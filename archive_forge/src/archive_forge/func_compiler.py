import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def compiler():
    """Get compiler to use for C++ to binary process. The precedence for
    choosing the compiler is as follows::

      1. `CXX` environment variable
      2. User configuration (~/.pythranrc)

    Returns None if none is set or if it's set to the empty string

    """
    cfg_cxx = str(cfg.get('compiler', 'CXX'))
    if not cfg_cxx:
        cfg_cxx = None
    return os.environ.get('CXX', cfg_cxx) or None