from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def can_linker_accept_rsp(self) -> bool:
    return is_windows()