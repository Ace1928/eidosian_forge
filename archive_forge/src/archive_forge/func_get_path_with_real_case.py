from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def get_path_with_real_case(filename):
    if filename.startswith('<') or not os_path_exists(filename):
        return filename
    parts = filename.lower().split('/')
    found = ''
    while parts and parts[0] == '':
        found += '/'
        parts = parts[1:]
    return _resolve_listing_parts(found, parts, filename)