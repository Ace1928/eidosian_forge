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
@staticmethod
def can_compile_remove_sources(compiler: 'Compiler', sources: T.List['FileOrString']) -> bool:
    removed = False
    for s in sources[:]:
        if compiler.can_compile(s):
            sources.remove(s)
            removed = True
    return removed