from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
def reformat_code(code):
    """
    Removes any leading 
 and dedents.
    """
    if code.startswith('\n'):
        code = code[1:]
    return dedent(code)