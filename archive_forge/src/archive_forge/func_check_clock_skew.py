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
def check_clock_skew(self, file_list: T.Iterable[str]) -> None:
    import time
    now = time.time()
    for f in file_list:
        absf = os.path.join(self.environment.get_build_dir(), f)
        ftime = os.path.getmtime(absf)
        delta = ftime - now
        if delta > 0.001:
            raise MesonException(f'Clock skew detected. File {absf} has a time stamp {delta:.4f}s in the future.')