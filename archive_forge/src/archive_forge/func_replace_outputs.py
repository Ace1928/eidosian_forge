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
def replace_outputs(self, args: T.List[str], private_dir: str, output_list: T.List[str]) -> T.List[str]:
    newargs: T.List[str] = []
    regex = re.compile('@OUTPUT(\\d+)@')
    for arg in args:
        m = regex.search(arg)
        while m is not None:
            index = int(m.group(1))
            src = f'@OUTPUT{index}@'
            arg = arg.replace(src, os.path.join(private_dir, output_list[index]))
            m = regex.search(arg)
        newargs.append(arg)
    return newargs