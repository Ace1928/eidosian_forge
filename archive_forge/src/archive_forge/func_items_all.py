import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def items_all(self, section_name: str) -> List[Tuple[str, List[str]]]:
    """:return: list((option, [values...]), ...) pairs of all items in the given section"""
    rv = _OMD(self._defaults)
    for k, vs in self._sections[section_name].items_all():
        if k == '__name__':
            continue
        if k in rv and rv.getall(k) == vs:
            continue
        for v in vs:
            rv.add(k, v)
    return rv.items_all()