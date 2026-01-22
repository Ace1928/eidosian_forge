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
def report_exception(exc_info, err_file):
    """Report an exception to err_file (typically stderr) and to brz.log.

    This will show either a full traceback or a short message as appropriate.

    :return: The appropriate exit code for this error.
    """
    log_exception_quietly()
    if 'error' in debug.debug_flags:
        print_exception(exc_info, err_file)
        return errors.EXIT_ERROR
    exc_type, exc_object, exc_tb = exc_info
    if isinstance(exc_object, KeyboardInterrupt):
        err_file.write('brz: interrupted\n')
        return errors.EXIT_ERROR
    elif isinstance(exc_object, MemoryError):
        err_file.write('brz: out of memory\n')
        if 'mem_dump' in debug.debug_flags:
            _dump_memory_usage(err_file)
        else:
            err_file.write('Use -Dmem_dump to dump memory to a file.\n')
        return errors.EXIT_ERROR
    elif isinstance(exc_object, ImportError) and str(exc_object).startswith('No module named '):
        report_user_error(exc_info, err_file, 'You may need to install this Python library separately.')
        return errors.EXIT_ERROR
    elif not getattr(exc_object, 'internal_error', True):
        report_user_error(exc_info, err_file)
        return errors.EXIT_ERROR
    elif isinstance(exc_object, EnvironmentError):
        if getattr(exc_object, 'errno', None) == errno.EPIPE:
            err_file.write('brz: broken pipe\n')
            return errors.EXIT_ERROR
        report_user_error(exc_info, err_file)
        return errors.EXIT_ERROR
    else:
        report_bug(exc_info, err_file)
        return errors.EXIT_INTERNAL_ERROR