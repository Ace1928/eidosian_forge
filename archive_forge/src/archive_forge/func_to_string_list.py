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
def to_string_list(self, sourcedir: str, builddir: str) -> T.List[str]:
    """Convert IncludeDirs object to a list of strings.

        :param sourcedir: The absolute source directory
        :param builddir: The absolute build directory, option, build dir will not
            be added if this is unset
        :returns: A list of strings (without compiler argument)
        """
    strlist: T.List[str] = []
    for idir in self.incdirs:
        strlist.append(os.path.join(sourcedir, self.curdir, idir))
        strlist.append(os.path.join(builddir, self.curdir, idir))
    return strlist