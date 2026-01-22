from __future__ import (absolute_import, division, print_function)
import keyword
import random
import uuid
from collections.abc import MutableMapping, MutableSequence
from json import dumps
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.parsing.splitter import parse_kv
def merge_hash(x, y, recursive=True, list_merge='replace'):
    """
    Return a new dictionary result of the merges of y into x,
    so that keys from y take precedence over keys from x.
    (x and y aren't modified)
    """
    if list_merge not in ('replace', 'keep', 'append', 'prepend', 'append_rp', 'prepend_rp'):
        raise AnsibleError("merge_hash: 'list_merge' argument can only be equal to 'replace', 'keep', 'append', 'prepend', 'append_rp' or 'prepend_rp'")
    _validate_mutable_mappings(x, y)
    if x == {} or x == y:
        return y.copy()
    if y == {}:
        return x
    x = x.copy()
    if not recursive and list_merge == 'replace':
        x.update(y)
        return x
    for key, y_value in y.items():
        if key not in x:
            x[key] = y_value
            continue
        x_value = x[key]
        if isinstance(x_value, MutableMapping) and isinstance(y_value, MutableMapping):
            if recursive:
                x[key] = merge_hash(x_value, y_value, recursive, list_merge)
            else:
                x[key] = y_value
            continue
        if isinstance(x_value, MutableSequence) and isinstance(y_value, MutableSequence):
            if list_merge == 'replace':
                x[key] = y_value
            elif list_merge == 'append':
                x[key] = x_value + y_value
            elif list_merge == 'prepend':
                x[key] = y_value + x_value
            elif list_merge == 'append_rp':
                x[key] = [z for z in x_value if z not in y_value] + y_value
            elif list_merge == 'prepend_rp':
                x[key] = y_value + [z for z in x_value if z not in y_value]
            continue
        x[key] = y_value
    return x