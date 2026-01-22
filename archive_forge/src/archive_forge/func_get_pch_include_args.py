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
def get_pch_include_args(self, compiler: 'Compiler', target: build.BuildTarget) -> T.List[str]:
    args: T.List[str] = []
    pchpath = self.get_target_private_dir(target)
    includeargs = compiler.get_include_args(pchpath, False)
    p = target.get_pch(compiler.get_language())
    if p:
        args += compiler.get_pch_use_args(pchpath, p[0])
    return includeargs + args