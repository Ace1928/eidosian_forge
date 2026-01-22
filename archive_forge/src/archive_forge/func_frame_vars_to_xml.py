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
def frame_vars_to_xml(frame_f_locals, hidden_ns=None):
    """ dumps frame variables to XML
    <var name="var_name" scope="local" type="type" value="value"/>
    """
    xml = []
    keys = sorted(frame_f_locals)
    return_values_xml = []
    for k in keys:
        try:
            v = frame_f_locals[k]
            eval_full_val = should_evaluate_full_value(v)
            if k == '_pydev_stop_at_break':
                continue
            if k == RETURN_VALUES_DICT:
                for name, val in v.items():
                    return_values_xml.append(var_to_xml(val, name, additional_in_xml=' isRetVal="True"'))
            elif hidden_ns is not None and k in hidden_ns:
                xml.append(var_to_xml(v, str(k), additional_in_xml=' isIPythonHidden="True"', evaluate_full_value=eval_full_val))
            else:
                xml.append(var_to_xml(v, str(k), evaluate_full_value=eval_full_val))
        except Exception:
            pydev_log.exception('Unexpected error, recovered safely.')
    return_values_xml.extend(xml)
    return ''.join(return_values_xml)