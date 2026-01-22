import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
@contextlib.contextmanager
def block_overlapping_deprecation(match_regex: str):
    """Context to block deprecation warnings raised within it.

    Useful if a function call might raise more than one warning,
    where only one warning is desired.

    Args:
        match_regex: DeprecationWarnings with message fields matching
            match_regex will be blocked.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=DeprecationWarning, message=f'(.|\n)*{match_regex}(.|\n)*')
        yield