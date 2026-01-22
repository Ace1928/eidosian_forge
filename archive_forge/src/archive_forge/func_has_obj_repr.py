import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def has_obj_repr(t):
    r = t.__repr__
    try:
        return obj_repr == r
    except Exception:
        return obj_repr is r