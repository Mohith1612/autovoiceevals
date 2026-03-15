#!/usr/bin/env python3
"""AutoVoiceEvals — entry point.

Usage:
    python main.py research [--resume] [--config config.yaml]
    python main.py pipeline [--config config.yaml]
"""

from autovoiceevals.cli import main

if __name__ == "__main__":
    main()
