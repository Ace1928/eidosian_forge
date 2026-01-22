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
def determine_linker_and_stdlib_args(self, target: build.BuildTarget) -> T.Tuple[T.Union['Compiler', 'StaticLinker'], T.List[str]]:
    """
        If we're building a static library, there is only one static linker.
        Otherwise, we query the target for the dynamic linker.
        """
    if isinstance(target, build.StaticLibrary):
        return (self.build.static_linker[target.for_machine], [])
    l, stdlib_args = target.get_clink_dynamic_linker_and_stdlibs()
    return (l, stdlib_args)