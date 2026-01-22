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
@lru_cache(maxsize=None)
def get_transitive_link_deps_mapping(self, prefix: str) -> T.Mapping[str, str]:
    result: T.Dict[str, str] = {}
    for i in self.link_targets:
        mapping = i.get_link_deps_mapping(prefix)
        result_tmp = mapping.copy()
        result_tmp.update(result)
        result = result_tmp
    return result