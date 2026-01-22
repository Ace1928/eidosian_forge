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
def get_target_depend_files(self, target: T.Union[build.CustomTarget, build.BuildTarget], absolute_paths: bool=False) -> T.List[str]:
    deps: T.List[str] = []
    for i in target.depend_files:
        if isinstance(i, mesonlib.File):
            if absolute_paths:
                deps.append(i.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir()))
            else:
                deps.append(i.rel_to_builddir(self.build_to_src))
        elif absolute_paths:
            deps.append(os.path.join(self.environment.get_source_dir(), target.subdir, i))
        else:
            deps.append(os.path.join(self.build_to_src, target.subdir, i))
    return deps