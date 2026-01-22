import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def get_stack_length(self, func):
    length = self._stack_length.get(func.__name__, _get_stack_length(func))
    return length