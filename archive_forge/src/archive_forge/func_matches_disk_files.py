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
def matches_disk_files(new_files: dict[str, str]) -> bool:
    for new_path, new_source in new_files.items():
        if not os.path.exists(new_path):
            return False
        with open(new_path, encoding='utf-8') as old_file:
            old_source = old_file.read()
        if old_source != new_source:
            return False
    return True