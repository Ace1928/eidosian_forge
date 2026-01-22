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
def get_external_rpath_dirs(self, target: build.BuildTarget) -> T.Set[str]:
    args: T.List[str] = []
    for lang in LANGUAGES_USING_LDFLAGS:
        try:
            e = self.environment.coredata.get_external_link_args(target.for_machine, lang)
            if isinstance(e, str):
                args.append(e)
            else:
                args.extend(e)
        except Exception:
            pass
    return self.get_rpath_dirs_from_link_args(args)