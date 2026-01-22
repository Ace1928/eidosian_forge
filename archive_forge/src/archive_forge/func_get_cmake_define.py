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