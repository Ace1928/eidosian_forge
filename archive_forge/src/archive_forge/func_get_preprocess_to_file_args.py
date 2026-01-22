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
def get_preprocess_to_file_args(self) -> T.List[str]:
    lang = _LANG_MAP.get(self.language, 'assembler-with-cpp')
    return self.get_preprocess_only_args() + [f'-x{lang}']