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
@property
def modes(self) -> str:
    modes = ''
    if self.can_read:
        modes += 'r'
    if self.can_write:
        modes += 'w'
    return modes