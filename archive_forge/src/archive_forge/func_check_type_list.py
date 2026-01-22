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
def check_type_list(value):
    """Verify that the value is a list or convert to a list

    A comma separated string will be split into a list. Raises a :class:`TypeError`
    if unable to convert to a list.

    :arg value: Value to validate or convert to a list

    :returns: Original value if it is already a list, single item list if a
        float, int, or string without commas, or a multi-item list if a
        comma-delimited string.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, string_types):
        return value.split(',')
    elif isinstance(value, int) or isinstance(value, float):
        return [str(value)]
    raise TypeError('%s cannot be converted to a list' % type(value))