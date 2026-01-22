from __future__ import annotations
import os
import typing as T
from ...mesonlib import EnvironmentException, OptionKey
def depfile_for_object(self, objfile: str) -> T.Optional[str]:
    return os.path.splitext(objfile)[0] + '.' + self.get_depfile_suffix()