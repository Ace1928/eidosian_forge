from __future__ import absolute_import, division, print_function
import logging
from functools import wraps, update_wrapper
import types
from warnings import warn
from passlib.utils.compat import PY3
@property
def im_func(self):
    """py2 alias"""
    return self.__func__