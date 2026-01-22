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
def do_conf_file(src: str, dst: str, confdata: 'ConfigurationData', variable_format: Literal['meson', 'cmake', 'cmake@'], encoding: str='utf-8', subproject: T.Optional[SubProject]=None) -> T.Tuple[T.Set[str], bool]:
    try:
        with open(src, encoding=encoding, newline='') as f:
            data = f.readlines()
    except Exception as e:
        raise MesonException(f'Could not read input file {src}: {e!s}')
    result, missing_variables, confdata_useless = do_conf_str(src, data, confdata, variable_format, subproject)
    dst_tmp = dst + '~'
    try:
        with open(dst_tmp, 'w', encoding=encoding, newline='') as f:
            f.writelines(result)
    except Exception as e:
        raise MesonException(f'Could not write output file {dst}: {e!s}')
    shutil.copymode(src, dst_tmp)
    replace_if_different(dst, dst_tmp)
    return (missing_variables, confdata_useless)