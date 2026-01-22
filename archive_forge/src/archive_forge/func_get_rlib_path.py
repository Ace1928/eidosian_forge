import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_rlib_path(r_home: str, system: str) -> str:
    """Get the path for the R shared library."""
    if system == 'FreeBSD' or system == 'Linux':
        lib_path = os.path.join(r_home, 'lib', 'libR.so')
    elif system == 'Darwin':
        lib_path = os.path.join(r_home, 'lib', 'libR.dylib')
    elif system == 'Windows':
        os.environ['PATH'] = os.pathsep.join((os.environ['PATH'], os.path.join(r_home, 'bin', r_version_folder)))
        lib_path = os.path.join(r_home, 'bin', r_version_folder, 'R.dll')
    else:
        raise ValueError('The system {system} is currently not supported.'.format(system=system))
    return lib_path