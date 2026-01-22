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
def flatten_object_list(self, target: build.BuildTarget, proj_dir_to_build_root: str='') -> T.Tuple[T.List[str], T.List[build.BuildTargetTypes]]:
    obj_list, deps = self._flatten_object_list(target, target.get_objects(), proj_dir_to_build_root)
    return (list(dict.fromkeys(obj_list)), deps)