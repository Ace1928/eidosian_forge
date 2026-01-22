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
def path_is_in_root(path: Path, root: Path, resolve: bool=False) -> bool:
    try:
        if resolve:
            path.resolve().relative_to(root.resolve())
        else:
            path.relative_to(root)
    except ValueError:
        return False
    return True