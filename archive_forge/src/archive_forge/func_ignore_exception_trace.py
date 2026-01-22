from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def ignore_exception_trace(trace):
    while trace is not None:
        filename = trace.tb_frame.f_code.co_filename
        if filename in ('<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>'):
            return True
        for file in FILES_WITH_IMPORT_HOOKS:
            if filename.endswith(file):
                return True
        trace = trace.tb_next
    return False