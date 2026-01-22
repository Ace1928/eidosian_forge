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
def get_custom_target_dir_include_args(self, target: build.CustomTarget, compiler: 'Compiler', *, absolute_path: bool=False) -> T.List[str]:
    incs: T.List[str] = []
    for i in self.get_custom_target_dirs(target, compiler, absolute_path=absolute_path):
        incs += compiler.get_include_args(i, False)
    return incs