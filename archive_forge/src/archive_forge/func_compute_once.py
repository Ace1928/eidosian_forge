import logging
import os
import torch
from . import _cpp_lib
from .checkpoint import (  # noqa: E402, F401
def compute_once(func):
    value = None

    def func_wrapper():
        nonlocal value
        if value is None:
            value = func()
        return value
    return func_wrapper