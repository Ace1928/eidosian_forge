from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def get_used_stdlib_args(self, link_language: str) -> T.List[str]:
    all_compilers = self.environment.coredata.compilers[self.for_machine]
    all_langs = set(self.compilers).union(self.get_langs_used_by_deps())
    stdlib_args: T.List[str] = []
    for dl in all_langs:
        if dl != link_language and (dl, link_language) not in self._MASK_LANGS:
            stdlib_args.extend(all_compilers[dl].language_stdlib_only_link_flags(self.environment))
    return stdlib_args