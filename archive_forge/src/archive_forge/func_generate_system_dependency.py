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
def generate_system_dependency(self, include_type: str) -> 'Dependency':
    new_dep = copy.deepcopy(self)
    new_dep.include_type = self._process_include_type_kw({'include_type': include_type})
    return new_dep