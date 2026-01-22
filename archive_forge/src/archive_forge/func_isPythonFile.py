import ast
import os
import platform
import re
import sys
from pyflakes import checker, __version__
from pyflakes import reporter as modReporter
def isPythonFile(filename):
    """Return True if filename points to a Python file."""
    if filename.endswith('.py'):
        return True
    if filename.endswith('~'):
        return False
    max_bytes = 128
    try:
        with open(filename, 'rb') as f:
            text = f.read(max_bytes)
            if not text:
                return False
    except OSError:
        return False
    return PYTHON_SHEBANG_REGEX.match(text)