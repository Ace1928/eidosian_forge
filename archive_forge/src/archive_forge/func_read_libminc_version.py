import os
import os.path
import warnings
from ..base import CommandLine
def read_libminc_version(s):
    if 'libminc' in s:
        return s.split(':')[1].strip()
    return None