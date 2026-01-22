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
def check_type_dict(value):
    """Verify that value is a dict or convert it to a dict and return it.

    Raises :class:`TypeError` if unable to convert to a dict

    :arg value: Dict or string to convert to a dict. Accepts ``k1=v2, k2=v2``.

    :returns: value converted to a dictionary
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, string_types):
        if value.startswith('{'):
            try:
                return json.loads(value)
            except Exception:
                result, exc = safe_eval(value, dict(), include_exceptions=True)
                if exc is not None:
                    raise TypeError('unable to evaluate string as dictionary')
                return result
        elif '=' in value:
            fields = []
            field_buffer = []
            in_quote = False
            in_escape = False
            for c in value.strip():
                if in_escape:
                    field_buffer.append(c)
                    in_escape = False
                elif c == '\\':
                    in_escape = True
                elif not in_quote and c in ("'", '"'):
                    in_quote = c
                elif in_quote and in_quote == c:
                    in_quote = False
                elif not in_quote and c in (',', ' '):
                    field = ''.join(field_buffer)
                    if field:
                        fields.append(field)
                    field_buffer = []
                else:
                    field_buffer.append(c)
            field = ''.join(field_buffer)
            if field:
                fields.append(field)
            return dict((x.split('=', 1) for x in fields))
        else:
            raise TypeError('dictionary requested, could not parse JSON or key=value')
    raise TypeError('%s cannot be converted to a dict' % type(value))