from __future__ import absolute_import, division, print_function
import os
import re
from ast import literal_eval
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._json_compat import json
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import (
def check_type_str(value, allow_conversion=True, param=None, prefix=''):
    """Verify that the value is a string or convert to a string.

    Since unexpected changes can sometimes happen when converting to a string,
    ``allow_conversion`` controls whether or not the value will be converted or a
    TypeError will be raised if the value is not a string and would be converted

    :arg value: Value to validate or convert to a string
    :arg allow_conversion: Whether to convert the string and return it or raise
        a TypeError

    :returns: Original value if it is a string, the value converted to a string
        if allow_conversion=True, or raises a TypeError if allow_conversion=False.
    """
    if isinstance(value, string_types):
        return value
    if allow_conversion and value is not None:
        return to_native(value, errors='surrogate_or_strict')
    msg = "'{0!r}' is not a string and conversion is not allowed".format(value)
    raise TypeError(to_native(msg))