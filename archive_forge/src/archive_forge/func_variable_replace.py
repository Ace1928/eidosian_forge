from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def variable_replace(match: T.Match[str]) -> str:
    if match.group(0).endswith('\\'):
        num_escapes = match.end(0) - match.start(0)
        return '\\' * (num_escapes // 2)
    elif match.group(0) == backslash_tag:
        return start_tag
    else:
        varname = match.group(1)
        var_str = ''
        if varname in confdata:
            var, _ = confdata.get(varname)
            if isinstance(var, str):
                var_str = var
            elif variable_format.startswith('cmake') and isinstance(var, bool):
                var_str = str(int(var))
            elif isinstance(var, int):
                var_str = str(var)
            else:
                msg = f'Tried to replace variable {varname!r} value with something other than a string or int: {var!r}'
                raise MesonException(msg)
        else:
            missing_variables.add(varname)
        return var_str