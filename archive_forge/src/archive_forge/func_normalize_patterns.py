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
def normalize_patterns(strings):
    if strings is None:
        strings = []
    elif isinstance(strings, (str, bytes)):
        strings = [strings]
    else:
        strings = list(strings)
    return strings