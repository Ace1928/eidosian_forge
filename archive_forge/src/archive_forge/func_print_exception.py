from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def print_exception(exc_info, err_file):
    import traceback
    exc_type, exc_object, exc_tb = exc_info
    err_file.write('brz: ERROR: {}: {}\n'.format(_qualified_exception_name(exc_type), exc_object))
    err_file.write('\n')
    traceback.print_exception(exc_type, exc_object, exc_tb, file=err_file)