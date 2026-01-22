import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def get_ioformat(name: str) -> IOFormat:
    """Return ioformat object or raise appropriate error."""
    if name not in ioformats:
        raise UnknownFileTypeError(name)
    fmt = ioformats[name]
    fmt.module
    return ioformats[name]