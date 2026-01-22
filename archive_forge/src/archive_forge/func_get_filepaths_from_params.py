from __future__ import print_function
import sys
import os
import platform
import io
import getopt
import re
import string
import errno
import copy
import glob
from jsbeautifier.__version__ import __version__
from jsbeautifier.javascript.options import BeautifierOptions
from jsbeautifier.javascript.beautifier import Beautifier
def get_filepaths_from_params(filepath_params, replace):
    filepaths = []
    if not filepath_params or (len(filepath_params) == 1 and filepath_params[0] == '-'):
        filepath_params = []
        filepaths.append('-')
    for filepath_param in filepath_params:
        if '-' == filepath_param:
            continue
        if os.path.isfile(filepath_param):
            filepaths.append(filepath_param)
        elif '*' in filepath_param or '?' in filepath_param:
            if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor <= 4):
                if '**' in filepath_param:
                    raise Exception('Recursive globs not supported on Python <= 3.4.')
                filepaths.extend(glob.glob(filepath_param))
            else:
                filepaths.extend(glob.glob(filepath_param, recursive=True))
        else:
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), filepath_param)
    if len(filepaths) > 1:
        replace = True
    elif filepaths and filepaths[0] == '-':
        replace = False
    filepaths = set(filepaths)
    return (filepaths, replace)