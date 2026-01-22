import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def cmdclass(values: Dict[str, str], package_dir: Optional[Mapping[str, str]]=None, root_dir: Optional[_Path]=None) -> Dict[str, Callable]:
    """Given a dictionary mapping command names to strings for qualified class
    names, apply :func:`resolve_class` to the dict values.
    """
    return {k: resolve_class(v, package_dir, root_dir) for k, v in values.items()}