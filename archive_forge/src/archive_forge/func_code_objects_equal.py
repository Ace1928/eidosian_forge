from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def code_objects_equal(code0, code1):
    for d in dir(code0):
        if d.startswith('_') or 'line' in d or d in ('replace', 'co_positions', 'co_qualname'):
            continue
        if getattr(code0, d) != getattr(code1, d):
            return False
    return True