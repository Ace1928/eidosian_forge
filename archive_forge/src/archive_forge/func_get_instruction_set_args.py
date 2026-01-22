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
def get_instruction_set_args(self, instruction_set: str) -> T.Optional[T.List[str]]:
    return gnulike_instruction_set_args.get(instruction_set, None)