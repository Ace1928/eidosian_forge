from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
@lru_cache(maxsize=None)
def rel_to_builddir(self, build_to_src: str) -> str:
    if self.is_built:
        return self.relative_name()
    else:
        return os.path.join(build_to_src, self.subdir, self.fname)