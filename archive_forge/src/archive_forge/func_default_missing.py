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
def default_missing(self) -> 'PerThreeMachine[T.Optional[_T]]':
    """Default host to build and target to host.

        This allows just specifying nothing in the native case, just host in the
        cross non-compiler case, and just target in the native-built
        cross-compiler case.
        """
    freeze = PerThreeMachine(self.build, self.host, self.target)
    if freeze.host is None:
        freeze.host = freeze.build
    if freeze.target is None:
        freeze.target = freeze.host
    return freeze