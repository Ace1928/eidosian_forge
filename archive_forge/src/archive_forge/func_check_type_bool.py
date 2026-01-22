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
def check_type_bool(value):
    """Verify that the value is a bool or convert it to a bool and return it.

    Raises :class:`TypeError` if unable to convert to a bool

    :arg value: String, int, or float to convert to bool. Valid booleans include:
         '1', 'on', 1, '0', 0, 'n', 'f', 'false', 'true', 'y', 't', 'yes', 'no', 'off'

    :returns: Boolean True or False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, string_types) or isinstance(value, (int, float)):
        return boolean(value)
    raise TypeError('%s cannot be converted to a bool' % type(value))