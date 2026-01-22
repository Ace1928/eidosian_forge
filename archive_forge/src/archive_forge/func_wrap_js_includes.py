from __future__ import annotations
import os.path
import typing as T
from ... import coredata
from ... import mesonlib
from ...mesonlib import OptionKey
from ...mesonlib import LibType
from mesonbuild.compilers.compilers import CompileCheckMode
def wrap_js_includes(args: T.List[str]) -> T.List[str]:
    final_args: T.List[str] = []
    for i in args:
        if i.endswith('.js') and (not i.startswith('-')):
            final_args += ['--js-library', i]
        else:
            final_args += [i]
    return final_args