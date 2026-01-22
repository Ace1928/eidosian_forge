import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
Return a list with all modules in `path`, which should be a list of
        directory names. If path is not given, sys.path will be used.