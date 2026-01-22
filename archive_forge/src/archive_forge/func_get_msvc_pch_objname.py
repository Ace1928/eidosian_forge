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
def get_msvc_pch_objname(self, lang: str, pch: T.List[str]) -> str:
    if len(pch) == 1:
        return f'meson_pch-{lang}.obj'
    return os.path.splitext(pch[1])[0] + '.obj'