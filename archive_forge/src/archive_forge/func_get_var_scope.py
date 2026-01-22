from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values):
    if attr_name.startswith("'"):
        if attr_name.endswith("'"):
            return ''
        else:
            i = attr_name.find("__' (")
            if i >= 0:
                attr_name = attr_name[1:i + 2]
    if handle_return_values and attr_name == RETURN_VALUES_DICT:
        return ''
    elif attr_name == GENERATED_LEN_ATTR_NAME:
        return ''
    if attr_name.startswith('__') and attr_name.endswith('__'):
        return DAPGrouper.SCOPE_SPECIAL_VARS
    if attr_name.startswith('_') or attr_name.endswith('__'):
        return DAPGrouper.SCOPE_PROTECTED_VARS
    try:
        if inspect.isroutine(attr_value) or isinstance(attr_value, MethodWrapperType):
            return DAPGrouper.SCOPE_FUNCTION_VARS
        elif inspect.isclass(attr_value):
            return DAPGrouper.SCOPE_CLASS_VARS
    except:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 0:
            pydev_log.exception()
    return ''