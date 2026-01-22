from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
def get_partial_dependency(self, *, compile_args: bool=False, link_args: bool=False, links: bool=False, includes: bool=False, sources: bool=False) -> 'ExternalLibrary':
    new = copy.copy(self)
    if not link_args:
        new.link_args = []
    return new