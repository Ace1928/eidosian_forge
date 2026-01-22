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
def typeslistify(item: 'T.Union[_T, T.Sequence[_T]]', types: 'T.Union[T.Type[_T], T.Tuple[T.Type[_T]]]') -> T.List[_T]:
    """
    Ensure that type(@item) is one of @types or a
    list of items all of which are of type @types
    """
    if isinstance(item, types):
        item = T.cast('T.List[_T]', [item])
    if not isinstance(item, list):
        raise MesonException('Item must be a list or one of {!r}, not {!r}'.format(types, type(item)))
    for i in item:
        if i is not None and (not isinstance(i, types)):
            raise MesonException('List item must be one of {!r}, not {!r}'.format(types, type(i)))
    return item