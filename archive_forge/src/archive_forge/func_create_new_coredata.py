from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def create_new_coredata(self, options: coredata.SharedCMDOptions) -> None:
    self.coredata = coredata.CoreData(options, self.scratch_dir, mesonlib.get_meson_command())
    self.first_invocation = True