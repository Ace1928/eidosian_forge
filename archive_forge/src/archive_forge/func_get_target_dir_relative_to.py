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
def get_target_dir_relative_to(self, t: build.Target, o: build.Target) -> str:
    """Get a target dir relative to another target's directory"""
    target_dir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(t))
    othert_dir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(o))
    return os.path.relpath(target_dir, othert_dir)