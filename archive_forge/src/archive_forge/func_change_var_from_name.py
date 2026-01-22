from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def change_var_from_name(self, container, name, new_value):
    try:
        set().add(new_value)
    except:
        return None
    for item in container:
        if str(id(item)) == name:
            container.remove(item)
            container.add(new_value)
            return str(id(new_value))
    return None