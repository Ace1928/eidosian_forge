from __future__ import annotations
import functools
import inspect
import os
import os.path
import sys
import threading
import traceback
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import (
from coverage.debug import short_filename, short_stack
from coverage.types import (
def panopticon(*names: str | None) -> AnyCallable:
    """Decorate a function to log its calls, but not really."""

    def _decorator(meth: AnyCallable) -> AnyCallable:
        return meth
    return _decorator