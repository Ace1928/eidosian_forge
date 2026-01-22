from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional
def get_variable_details(val, evaluate_full_value=True, to_string=None, context: Optional[str]=None):
    """
    :param context:
        This is the context in which the variable is being requested. Valid values:
            "watch",
            "repl",
            "hover",
            "clipboard"
    """
    try:
        is_exception_on_eval = val.__class__ == ExceptionOnEvaluate
    except:
        is_exception_on_eval = False
    if is_exception_on_eval:
        v = val.result
    else:
        v = val
    _type, type_name, resolver = get_type(v)
    type_qualifier = getattr(_type, '__module__', '')
    if not evaluate_full_value:
        value = DEFAULT_VALUE
    else:
        try:
            str_from_provider = _str_from_providers(v, _type, type_name, context)
            if str_from_provider is not None:
                value = str_from_provider
            elif to_string is not None:
                value = to_string(v)
            elif hasattr_checked(v, '__class__'):
                if v.__class__ == frame_type:
                    value = pydevd_resolver.frameResolver.get_frame_name(v)
                elif v.__class__ in (list, tuple):
                    if len(v) > 300:
                        value = '%s: %s' % (str(v.__class__), '<Too big to print. Len: %s>' % (len(v),))
                    else:
                        value = '%s: %s' % (str(v.__class__), v)
                else:
                    try:
                        cName = str(v.__class__)
                        if cName.find('.') != -1:
                            cName = cName.split('.')[-1]
                        elif cName.find("'") != -1:
                            cName = cName[cName.index("'") + 1:]
                        if cName.endswith("'>"):
                            cName = cName[:-2]
                    except:
                        cName = str(v.__class__)
                    value = '%s: %s' % (cName, v)
            else:
                value = str(v)
        except:
            try:
                value = repr(v)
            except:
                value = 'Unable to get repr for %s' % v.__class__
    try:
        if value.__class__ == bytes:
            value = value.decode('utf-8', 'replace')
    except TypeError:
        pass
    return (type_name, type_qualifier, is_exception_on_eval, resolver, value)