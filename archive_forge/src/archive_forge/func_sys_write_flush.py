import os
import subprocess as sp
import sys
import warnings
import proglog
from .compat import DEVNULL
def sys_write_flush(s):
    """ Writes and flushes without delay a text in the console """
    sys.stdout.write(s)
    sys.stdout.flush()