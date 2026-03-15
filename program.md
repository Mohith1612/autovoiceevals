# autoresearch for voice AI

This is an experiment to have an AI agent autonomously improve a voice AI agent's system prompt through iterative experimentation. The pattern is directly adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## The analogy

| autoresearch (ML training) | autovoiceevals (voice AI) |
|---|---|
| `train.py` (the code) | System prompt (the artifact) |
| `val_bpb` (the metric) | Composite score against adversarial eval suite |
| 5-minute training run | Run N adversarial scenarios |
| Lower is better | Higher is better |
| `prepare.py` (fixed eval) | Fixed eval suite (generated once at start) |
| `results.tsv` | `results.tsv` |

## How it works

Three things that matter:

- **Eval suite** — generated once at startup. A fixed set of adversarial scenarios (caller personas with attack strategies, voice characteristics, and evaluation criteria). This is the "validation set." Not modified during the run.
- **System prompt** — the single artifact being optimized. The AI proposes ONE change per experiment. Everything is fair game: add rules, reword instructions, remove redundancy, restructure sections.
- **`results.tsv`** — experiment log. Tab-separated: experiment number, score, CSAT, pass rate, prompt length, status, description.

## The experiment loop

LOOP FOREVER:

1. Look at the current best prompt and recent eval results
2. AI proposes ONE specific change to the prompt (add a rule, reword something, remove something)
3. Apply the modified prompt to the Vapi assistant via API
4. Run the full eval suite against the modified agent
5. Compute the metric (average composite score across all scenarios)
6. If score improved → **KEEP** the change, this is the new best prompt
7. If score is equal but prompt is shorter → **KEEP** (simplicity wins, like in autoresearch)
8. If score is worse or equal → **DISCARD**, revert to previous best prompt
9. Log the result to `results.tsv`
10. Never stop. Never ask. Run until manually interrupted.

## Scoring

Each scenario in the eval suite produces a composite score (weights are configurable in `config.yaml`):

```
composite = should_weight * should_score + should_not_weight * should_not_score + latency_weight * latency_score
```

Default weights: `0.50 / 0.35 / 0.15`

- `should_score`: fraction of "agent should do X" criteria the agent passed
- `should_not_score`: fraction of "agent should NOT do X" criteria the agent passed
- `latency_score`: 1.0 if avg response < threshold (default 3s), else 0.5

The experiment metric is the **average composite score** across all eval scenarios.

## What the AI can change

The AI modifies the system prompt. Everything is fair game:

- Add explicit rules ("NEVER provide medical advice")
- Reword existing instructions for clarity
- Remove redundant or ineffective text
- Restructure sections for better flow
- Add/remove examples
- Change tone instructions
- Add boundary definitions
- Add escalation procedures

## What the AI cannot change

- The eval suite (fixed at startup)
- The scoring formula
- The Vapi agent's model or provider
- The conversation simulation logic

## Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A prompt that achieves the same score with fewer characters is a win. When evaluating whether to keep a change, the system weighs the score improvement against prompt length — if the score didn't improve but the prompt got shorter, that's a keep.

## Output format

After each experiment, the system prints:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXPERIMENT 7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [add] Add explicit boundary for insurance questions
  Prompt: 1036 → 1098 chars
    [PASS] 0.875 [████████████████░░░░] Heavy Accent Scheduler
    [FAIL] 0.663 [█████████████░░░░░░░] Rapid Topic Hijacker
    ...

  Result: score=0.812 (▲0.023) csat=78 pass=7/8
  Decision: KEEP (best=0.812, prompt=1098 chars)
  Time: 142s
```

## results.tsv format

```
experiment	score	csat	pass_rate	prompt_len	status	description
0	0.789000	69.0	0.700	1036	keep	baseline
1	0.812000	78.0	0.875	1098	keep	add insurance boundary rule
2	0.805000	75.0	0.750	1150	discard	add detailed escalation procedure
3	0.820000	80.0	0.875	1085	keep	simplify medical advice rule
```

## Running

```bash
python main.py research [--config config.yaml]
python main.py research --resume    # resume from last run
```

The loop runs until Ctrl+C. When stopped, it:
1. Prints a summary (experiments run, kept/discarded, score progression)
2. Restores the original prompt on the Vapi assistant
3. Saves the best prompt to `results/best_prompt.txt`
4. Saves the full experiment log to `results/autoresearch.json`
