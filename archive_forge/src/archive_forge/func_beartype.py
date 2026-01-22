import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
def beartype(func):
    return func