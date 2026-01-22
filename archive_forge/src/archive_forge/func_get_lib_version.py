import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Mapping, Union
def get_lib_version():
    try:
        libver = metadata.version('redis')
    except metadata.PackageNotFoundError:
        libver = '99.99.99'
    return libver