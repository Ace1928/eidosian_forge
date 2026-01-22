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
def get_testlike_targets(self, benchmark: bool=False) -> T.OrderedDict[str, T.Union[build.BuildTarget, build.CustomTarget]]:
    result: T.OrderedDict[str, T.Union[build.BuildTarget, build.CustomTarget]] = OrderedDict()
    targets = self.build.get_benchmarks() if benchmark else self.build.get_tests()
    for t in targets:
        exe = t.exe
        if isinstance(exe, (build.CustomTarget, build.BuildTarget)):
            result[exe.get_id()] = exe
        for arg in t.cmd_args:
            if not isinstance(arg, (build.CustomTarget, build.BuildTarget)):
                continue
            result[arg.get_id()] = arg
        for dep in t.depends:
            assert isinstance(dep, (build.CustomTarget, build.BuildTarget))
            result[dep.get_id()] = dep
    return result