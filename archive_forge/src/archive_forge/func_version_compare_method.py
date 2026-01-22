from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@typed_pos_args('str.version_compare', str)
def version_compare_method(self, args: T.Tuple[str], kwargs: TYPE_kwargs) -> bool:
    self.interpreter.tmp_meson_version = args[0]
    return version_compare(self.held_object, args[0])