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
def generate_custom_install_script(self, d: InstallData) -> None:
    d.install_scripts = self.build.install_scripts
    for i in d.install_scripts:
        if not i.tag:
            mlog.debug('Failed to guess install tag for install script:', ' '.join(i.cmd_args))