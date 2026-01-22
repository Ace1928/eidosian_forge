from __future__ import annotations
import os
import signal
import sys
import threading
from collections import deque
from typing import (
from wcwidth import wcwidth
def get_term_environment_variable() -> str:
    """Return the $TERM environment variable."""
    return os.environ.get('TERM', '')