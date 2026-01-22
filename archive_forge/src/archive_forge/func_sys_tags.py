import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def sys_tags(*, warn: bool=False) -> Iterator[Tag]:
    """
    Returns the sequence of tag triples for the running interpreter.

    The order of the sequence corresponds to priority order for the
    interpreter, from most to least important.
    """
    interp_name = interpreter_name()
    if interp_name == 'cp':
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags()
    if interp_name == 'pp':
        interp = 'pp3'
    elif interp_name == 'cp':
        interp = 'cp' + interpreter_version(warn=warn)
    else:
        interp = None
    yield from compatible_tags(interpreter=interp)