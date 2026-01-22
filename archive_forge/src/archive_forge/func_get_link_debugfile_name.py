from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_link_debugfile_name(self, targetfile: str) -> T.Optional[str]:
    return None