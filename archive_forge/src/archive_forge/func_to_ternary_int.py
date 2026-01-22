import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def to_ternary_int(val) -> int:
    """Convert a value to the ternary 1/0/-1 int used for True/None/False in
    attributes such as SENT_START: True/1/1.0 is 1 (True), None/0/0.0 is 0
    (None), any other values are -1 (False).
    """
    if val is True:
        return 1
    elif val is None:
        return 0
    elif val is False:
        return -1
    elif val == 1:
        return 1
    elif val == 0:
        return 0
    else:
        return -1