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
def load_config_from_str(text: str, overrides: Dict[str, Any]=SimpleFrozenDict(), interpolate: bool=False):
    """Load a full config from a string. Wrapper around Thinc's Config.from_str.

    text (str): The string config to load.
    interpolate (bool): Whether to interpolate and resolve variables.
    RETURNS (Config): The loaded config.
    """
    return Config(section_order=CONFIG_SECTION_ORDER).from_str(text, overrides=overrides, interpolate=interpolate)