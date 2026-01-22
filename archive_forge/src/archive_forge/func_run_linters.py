from __future__ import annotations
import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING
import attrs
import astor
from __future__ import annotations
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import GLOBAL_RUN_CONTEXT
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING
from outcome import Outcome
import contextvars
from ._run import _NO_SEND, RunStatistics, Task
from ._entry_queue import TrioToken
from .._abc import Clock
from ._instrumentation import Instrument
from typing import TYPE_CHECKING
from typing import Callable, ContextManager, TYPE_CHECKING
from typing import TYPE_CHECKING, ContextManager
def run_linters(file: File, source: str) -> str:
    """Format the specified file using black and ruff.

    Returns:
      Formatted source code.

    Raises:
      ImportError: If either is not installed.
      SystemExit: If either failed.
    """
    success, response = run_black(file, source)
    if not success:
        print(response)
        sys.exit(1)
    success, response = run_ruff(file, response)
    if not success:
        print(response)
        sys.exit(1)
    return response