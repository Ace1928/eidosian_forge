import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def nargs_error(name, takes, given):
    """Generate a TypeError to be raised by function calls with wrong arity."""
    return TypeError(f'{name}() takes {takes} positional arguments but {given} were given')