from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def get_dependency_compile_args(self, dep: 'Dependency') -> T.List[str]:
    if dep.get_include_type() == 'system':
        converted: T.List[str] = []
        for i in dep.get_compile_args():
            if i.startswith('-isystem'):
                converted += ['/clang:' + i]
            else:
                converted += [i]
        return converted
    else:
        return dep.get_compile_args()