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
def do_define(regex: T.Pattern[str], line: str, confdata: 'ConfigurationData', variable_format: Literal['meson', 'cmake', 'cmake@'], subproject: T.Optional[SubProject]=None) -> str:
    cmake_bool_define = False
    if variable_format != 'meson':
        cmake_bool_define = 'cmakedefine01' in line

    def get_cmake_define(line: str, confdata: 'ConfigurationData') -> str:
        arr = line.split()
        if cmake_bool_define:
            v, desc = confdata.get(arr[1])
            return str(int(bool(v)))
        define_value: T.List[str] = []
        for token in arr[2:]:
            try:
                v, _ = confdata.get(token)
                define_value += [str(v)]
            except KeyError:
                define_value += [token]
        return ' '.join(define_value)
    arr = line.split()
    if len(arr) != 2:
        if variable_format == 'meson':
            raise MesonException('#mesondefine does not contain exactly two tokens: %s' % line.strip())
        elif subproject is not None:
            from ..interpreterbase.decorators import FeatureNew
            FeatureNew.single_use('cmakedefine without exactly two tokens', '0.54.1', subproject)
    varname = arr[1]
    try:
        v, _ = confdata.get(varname)
    except KeyError:
        if cmake_bool_define:
            return '#define %s 0\n' % varname
        else:
            return '/* #undef %s */\n' % varname
    if isinstance(v, str) or variable_format != 'meson':
        if variable_format == 'meson':
            result = v
        else:
            if not cmake_bool_define and (not v):
                return '/* #undef %s */\n' % varname
            result = get_cmake_define(line, confdata)
        result = f'#define {varname} {result}'.strip() + '\n'
        result, _ = do_replacement(regex, result, variable_format, confdata)
        return result
    elif isinstance(v, bool):
        if v:
            return '#define %s\n' % varname
        else:
            return '#undef %s\n' % varname
    elif isinstance(v, int):
        return '#define %s %d\n' % (varname, v)
    else:
        raise MesonException('#mesondefine argument "%s" is of unknown type.' % varname)