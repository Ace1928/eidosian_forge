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
def patched_new(cls, *args, **kwargs):
    qualname = clazz.__qualname__ if name is None else name
    _warn_or_error(f'{qualname} was used but is deprecated.\nIt will be removed in cirq {deadline}.\n{fix}\n')
    return clazz_new(cls)