from __future__ import with_statement
import logging
import os
from textwrap import dedent
import time
import pathlib
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import weakref
from datetime import timedelta
from tokenize import open as open_py_source
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from .logger import Logger, format_time, pformat
from ._store_backends import StoreBackendBase, FileSystemStoreBackend
from ._store_backends import CacheWarning  # noqa
def check_call_in_cache(self, *args, **kwargs):
    """Check if function call is in the memory cache.

        Does not call the function or do any work besides func inspection
        and arg hashing.

        Returns
        -------
        is_call_in_cache: bool
            Whether or not the result of the function has been cached
            for the input arguments that have been passed.
        """
    func_id, args_id = self._get_output_identifiers(*args, **kwargs)
    return self.store_backend.contains_item((func_id, args_id))