import fnmatch
import glob
import os.path
import sys
from _pydev_bundle import pydev_log
import pydevd_file_utils
import json
from collections import namedtuple
from _pydev_bundle._pydev_saved_modules import threading
from pydevd_file_utils import normcase
from _pydevd_bundle.pydevd_constants import USER_CODE_BASENAMES_STARTING_WITH, \
from _pydevd_bundle import pydevd_constants
def glob_matches_path(path, pattern, sep=os.sep, altsep=os.altsep):
    if altsep:
        pattern = pattern.replace(altsep, sep)
        path = path.replace(altsep, sep)
    drive = ''
    if len(path) > 1 and path[1] == ':':
        drive, path = (path[0], path[2:])
    if drive and len(pattern) > 1:
        if pattern[1] == ':':
            if drive.lower() != pattern[0].lower():
                return False
            pattern = pattern[2:]
    patterns = pattern.split(sep)
    paths = path.split(sep)
    if paths:
        if paths[0] == '':
            paths = paths[1:]
    if patterns:
        if patterns[0] == '':
            patterns = patterns[1:]
    return _check_matches(patterns, paths)