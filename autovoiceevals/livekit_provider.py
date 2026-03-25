"""LiveKit provider — Phase 1 (data messages).

Runs adversarial eval conversations via a LiveKit room using the data
channel (text messages), not audio. The caller bot joins the room,
sends turns as JSON data messages, and waits for the agent to respond
the same way.

Requirements
------------
* pip install "livekit>=1.0.0"
* LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env

Agent requirements
------------------
The target agent must listen on the configured data_topic and reply
on the same topic. Message format (JSON):

    Caller → Agent:  {"role": "user",      "content": "<turn text>"}
    Agent  → Caller: {"role": "assistant", "content": "<reply text>"}

Plain-text responses (non-JSON) are also accepted.

Prompt management
-----------------
If agent_backend is "smallest", prompt reads/writes are delegated to
SmallestClient. If "none", get_system_prompt/update_prompt will raise
NotImplementedError — use the livekit provider only for conversations
and manage prompts externally.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from .models import Turn, Conversation

DEFAULT_END_PHRASES = [
    "have a great day",
    "goodbye",
    "talk to you soon",
    "take care",
]


class LiveKitClient:
    """Voice platform client that uses LiveKit data channel messages."""

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        room_prefix: str = "eval",
        data_topic: str = "text",
        response_timeout: float = 30.0,
        agent_join_timeout: float = 30.0,
        end_phrases: list[str] | None = None,
        agent_backend=None,
    ):
        """
        Args:
            url:                  LiveKit server WebSocket URL (wss://...).
            api_key:              LiveKit API key.
            api_secret:           LiveKit API secret.
            room_prefix:          Prefix for generated room names.
            data_topic:           Data channel topic for messages.
            response_timeout:     Seconds to wait for each agent response.
            agent_join_timeout:   Seconds to wait for the agent to join.
            end_phrases:          Phrases that signal conversation end.
            agent_backend:        Optional client for prompt management
                                  (e.g. SmallestClient). If None, prompt
                                  methods raise NotImplementedError.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_prefix = room_prefix
        self.data_topic = data_topic
        self.response_timeout = response_timeout
        self.agent_join_timeout = agent_join_timeout
        self.end_phrases = end_phrases or DEFAULT_END_PHRASES
        self._backend = agent_backend

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def run_conversation(
        self,
        assistant_id: str,
        scenario_id: str,
        caller_turns: list[str],
        max_turns: int = 12,
        scenario=None,
        dynamic_variables: dict | None = None,
        simulate_timeout_secs: int | None = None,
    ) -> Conversation:
        """Run a multi-turn conversation via LiveKit data messages.

        Each call creates a unique room. The agent is expected to join
        that room (via webhook dispatch or a fixed room name configured
        on the agent side) and respond to data messages on data_topic.

        The ``scenario``, ``dynamic_variables``, and ``simulate_timeout_secs``
        parameters are accepted for interface compatibility but are unused in
        data-channel mode.
        """
        return asyncio.run(
            self._run_async(assistant_id, scenario_id, caller_turns, max_turns)
        )

    async def _run_async(
        self,
        assistant_id: str,
        scenario_id: str,
        caller_turns: list[str],
        max_turns: int,
    ) -> Conversation:
        try:
            from livekit import rtc
            from livekit.api import AccessToken, VideoGrants
        except ImportError:
            conv = Conversation(scenario_id=scenario_id)
            conv.error = (
                "livekit package not installed. "
                "Run: pip install 'livekit>=1.0.0'"
            )
            return conv

        conv = Conversation(scenario_id=scenario_id)
        total_latency = 0.0

        # Unique room per conversation to avoid cross-talk
        room_name = f"{self.room_prefix}-{scenario_id}-{uuid.uuid4().hex[:8]}"
        identity = f"caller-{uuid.uuid4().hex[:6]}"

        token = (
            AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()
        response_queue: asyncio.Queue[str] = asyncio.Queue()
        agent_joined = asyncio.Event()

        @room.on("data_received")
        def on_data(packet):
            # livekit >= 1.0: packet is a DataPacket with .data bytes attribute
            try:
                raw = bytes(packet.data) if hasattr(packet, "data") else bytes(packet)
                text = raw.decode("utf-8")
                response_queue.put_nowait(text)
            except Exception:
                pass

        @room.on("participant_connected")
        def on_participant(_participant):
            agent_joined.set()

        try:
            await room.connect(self.url, token)
        except Exception as e:
            conv.error = f"LiveKit connect failed: {str(e)[:200]}"
            return conv

        # Agent may already be in the room when we join
        if room.remote_participants:
            agent_joined.set()

        try:
            await asyncio.wait_for(
                agent_joined.wait(), timeout=self.agent_join_timeout
            )
        except asyncio.TimeoutError:
            conv.error = (
                f"Agent did not join room '{room_name}' "
                f"within {self.agent_join_timeout}s. "
                "Ensure the agent is configured to dispatch to this room."
            )
            await room.disconnect()
            return conv

        # Run turns
        for msg in caller_turns[:max_turns]:
            if not msg or not msg.strip():
                msg = "..."

            conv.turns.append(Turn(role="caller", content=msg))

            payload = json.dumps({"role": "user", "content": msg}).encode("utf-8")

            try:
                t0 = time.time()
                await room.local_participant.publish_data(
                    payload,
                    reliable=True,
                    topic=self.data_topic,
                )

                raw_response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=self.response_timeout,
                )
                latency = (time.time() - t0) * 1000

                # Accept JSON {"role": "assistant", "content": "..."} or plain text
                try:
                    parsed = json.loads(raw_response)
                    agent_msg = parsed.get("content", raw_response)
                except (json.JSONDecodeError, AttributeError):
                    agent_msg = raw_response

                conv.turns.append(
                    Turn(role="assistant", content=agent_msg, latency_ms=latency)
                )
                total_latency += latency

                if any(p in agent_msg.lower() for p in self.end_phrases):
                    break

            except asyncio.TimeoutError:
                conv.error = f"Response timeout (>{self.response_timeout}s)"
                break
            except Exception as e:
                conv.error = str(e)[:200]
                break

            await asyncio.sleep(0.1)

        await room.disconnect()

        n = len(conv.agent_turns)
        conv.avg_latency_ms = total_latency / n if n else 0
        return conv

    # ------------------------------------------------------------------
    # Prompt management (delegated to agent_backend)
    # ------------------------------------------------------------------

    def get_system_prompt(self, agent_id: str) -> str:
        """Read the current system prompt via the configured backend."""
        if self._backend is not None:
            return self._backend.get_system_prompt(agent_id)
        raise NotImplementedError(
            "No agent_backend configured for LiveKit provider. "
            "Set livekit.agent_backend: 'smallest' in config.yaml, "
            "or manage prompts externally."
        )

    def update_prompt(self, agent_id: str, new_prompt: str) -> bool:
        """Update the system prompt via the configured backend."""
        if self._backend is not None:
            return self._backend.update_prompt(agent_id, new_prompt)
        raise NotImplementedError(
            "No agent_backend configured for LiveKit provider. "
            "Set livekit.agent_backend: 'smallest' in config.yaml, "
            "or manage prompts externally."
        )
