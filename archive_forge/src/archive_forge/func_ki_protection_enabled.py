from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attrs
from .._util import is_main_thread
def ki_protection_enabled(frame: types.FrameType | None) -> bool:
    while frame is not None:
        if LOCALS_KEY_KI_PROTECTION_ENABLED in frame.f_locals:
            return bool(frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED])
        if frame.f_code.co_name == '__del__':
            return True
        frame = frame.f_back
    return True