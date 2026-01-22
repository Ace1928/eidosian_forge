import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def to_terminal(self) -> bool:
    """Return whether the output stream is a terminal."""
    return self._tty