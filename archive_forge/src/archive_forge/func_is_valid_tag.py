from __future__ import absolute_import
import os
import sys
import errno
from ..Compiler import Errors
from ..Compiler.StringEncoding import EncodedString
def is_valid_tag(name):
    """
    Names like '.0' are used internally for arguments
    to functions creating generator expressions,
    however they are not identifiers.

    See https://github.com/cython/cython/issues/5552
    """
    if isinstance(name, EncodedString):
        if name.startswith('.') and name[1:].isdecimal():
            return False
    return True