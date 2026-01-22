from __future__ import annotations
import os.path
import typing as T
from ... import coredata
from ... import mesonlib
from ...mesonlib import OptionKey
from ...mesonlib import LibType
from mesonbuild.compilers.compilers import CompileCheckMode
@classmethod
def native_args_to_unix(cls, args: T.List[str]) -> T.List[str]:
    return wrap_js_includes(super().native_args_to_unix(args))