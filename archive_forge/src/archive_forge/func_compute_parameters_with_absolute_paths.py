from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
    for idx, i in enumerate(parameter_list):
        if i[:2] == '-I' or i[:2] == '-L':
            parameter_list[idx] = i[:2] + os.path.normpath(os.path.join(build_dir, i[2:]))
    return parameter_list