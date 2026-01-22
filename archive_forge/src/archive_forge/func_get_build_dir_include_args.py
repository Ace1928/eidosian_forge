from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def get_build_dir_include_args(self, target: build.BuildTarget, compiler: 'Compiler', *, absolute_path: bool=False) -> T.List[str]:
    if absolute_path:
        curdir = os.path.join(self.build_dir, target.get_subdir())
    else:
        curdir = target.get_subdir()
        if curdir == '':
            curdir = '.'
    return compiler.get_include_args(curdir, False)