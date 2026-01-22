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
def is_unconstrained_version(constraint: str, prereleases: bool=True) -> Optional[bool]:
    if constraint[0].isdigit():
        return False
    try:
        spec = SpecifierSet(constraint)
    except InvalidSpecifier:
        return None
    spec.prereleases = prereleases
    specs = [sp for sp in spec]
    if len(specs) == 1 and specs[0].operator in ('>', '>='):
        return True
    if any((sp.operator in '==' for sp in specs)):
        return False
    has_upper = any((sp.operator in ('<', '<=') for sp in specs))
    has_lower = any((sp.operator in ('>', '>=') for sp in specs))
    if has_upper and has_lower:
        return False
    return True