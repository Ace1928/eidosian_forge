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
def generate_man_install(self, d: InstallData) -> None:
    manroot = self.environment.get_mandir()
    man = self.build.get_man()
    for m in man:
        for f in m.get_sources():
            num = f.split('.')[-1]
            subdir = m.get_custom_install_dir()
            if subdir is None:
                if m.locale:
                    subdir = os.path.join('{mandir}', m.locale, 'man' + num)
                else:
                    subdir = os.path.join('{mandir}', 'man' + num)
            fname = f.fname
            if m.locale:
                fname = fname.replace(f'.{m.locale}', '')
            srcabs = f.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
            dstname = os.path.join(subdir, os.path.basename(fname))
            dstabs = dstname.replace('{mandir}', manroot)
            i = InstallDataBase(srcabs, dstabs, dstname, m.get_custom_install_mode(), m.subproject, tag='man')
            d.man.append(i)