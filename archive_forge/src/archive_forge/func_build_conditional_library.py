from __future__ import annotations
import os
import sys
import threading
import types
import typing
import warnings
import cryptography
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import _openssl, openssl
from cryptography.hazmat.bindings.openssl._conditional import CONDITIONAL_NAMES
def build_conditional_library(lib: typing.Any, conditional_names: typing.Dict[str, typing.Callable[[], typing.List[str]]]) -> typing.Any:
    conditional_lib = types.ModuleType('lib')
    conditional_lib._original_lib = lib
    excluded_names = set()
    for condition, names_cb in conditional_names.items():
        if not getattr(lib, condition):
            excluded_names.update(names_cb())
    for attr in dir(lib):
        if attr not in excluded_names:
            setattr(conditional_lib, attr, getattr(lib, attr))
    return conditional_lib