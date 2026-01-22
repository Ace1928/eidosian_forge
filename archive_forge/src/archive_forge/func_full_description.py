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
def full_description(self) -> str:
    lines = [f'Name:        {self.name}', f'Description: {self.description}', f'Modes:       {self.modes}', f'Encoding:    {self.encoding}', f'Module:      {self.module_name}', f'Code:        {self.code}', f'Extensions:  {self.extensions}', f'Globs:       {self.globs}', f'Magic:       {self.magic}']
    return '\n'.join(lines)