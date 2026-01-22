import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@staticmethod
def dist_error(*args):
    """Raise a compiler error"""
    from distutils.errors import CompileError
    raise CompileError(_Distutils._dist_str(*args))