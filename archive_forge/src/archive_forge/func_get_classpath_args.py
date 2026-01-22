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
def get_classpath_args(self):
    cp_paths = [os.path.join(l.get_subdir(), l.get_filename()) for l in self.link_targets]
    cp_string = os.pathsep.join(cp_paths)
    if cp_string:
        return ['-cp', os.pathsep.join(cp_paths)]
    return []