"""pytest configuration: ensure project root is on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

# Project root is two levels up from this file (tests/conftest.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
