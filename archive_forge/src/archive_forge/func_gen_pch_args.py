from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def gen_pch_args(self, header: str, source: str, pchname: str) -> T.Tuple[str, T.List[str]]:
    objname = os.path.splitext(source)[0] + '.obj'
    return (objname, ['/Yc' + header, '/Fp' + pchname, '/Fo' + objname])