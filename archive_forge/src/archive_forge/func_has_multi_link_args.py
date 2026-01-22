from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def has_multi_link_args(self, args: T.List[str], env: 'Environment') -> T.Tuple[bool, bool]:
    return (False, False)