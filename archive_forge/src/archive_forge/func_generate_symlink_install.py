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
def generate_symlink_install(self, d: InstallData) -> None:
    links: T.List[build.SymlinkData] = self.build.get_symlinks()
    for l in links:
        assert isinstance(l, build.SymlinkData)
        install_dir = l.install_dir
        name_abs = os.path.join(install_dir, l.name)
        tag = l.install_tag or self.guess_install_tag(name_abs)
        s = InstallSymlinkData(l.target, name_abs, install_dir, l.subproject, tag)
        d.symlinks.append(s)