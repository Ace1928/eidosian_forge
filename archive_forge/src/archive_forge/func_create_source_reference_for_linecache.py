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
def create_source_reference_for_linecache(server_filename):
    source_reference = _next_source_reference()
    pydev_log.debug('Created linecache id source reference: %s for server filename: %s', source_reference, server_filename)
    _line_cache_source_reference_to_server_filename[source_reference] = server_filename
    return source_reference