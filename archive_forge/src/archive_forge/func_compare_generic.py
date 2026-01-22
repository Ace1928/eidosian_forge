from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def compare_generic(a, b, method, datatype):
    """
    Compare values a and b as described by method and datatype.

    Returns ``True`` if the values compare equal, and ``False`` if not.

    ``a`` is usually the module's parameter, while ``b`` is a property
    of the current object. ``a`` must not be ``None`` (except for
    ``datatype == 'value'``).

    Valid values for ``method`` are:
    - ``ignore`` (always compare as equal);
    - ``strict`` (only compare if really equal)
    - ``allow_more_present`` (allow b to have elements which a does not have).

    Valid values for ``datatype`` are:
    - ``value``: for simple values (strings, numbers, ...);
    - ``list``: for ``list``s or ``tuple``s where order matters;
    - ``set``: for ``list``s, ``tuple``s or ``set``s where order does not
      matter;
    - ``set(dict)``: for ``list``s, ``tuple``s or ``sets`` where order does
      not matter and which contain ``dict``s; ``allow_more_present`` is used
      for the ``dict``s, and these are assumed to be dictionaries of values;
    - ``dict``: for dictionaries of values.
    """
    if method == 'ignore':
        return True
    if a is None or b is None:
        if a == b:
            return True
        if datatype == 'value':
            return False
        if method == 'allow_more_present' and a is None:
            return True
        return len(b if a is None else a) == 0
    if datatype == 'value':
        return a == b
    elif datatype == 'list':
        if method == 'strict':
            return a == b
        else:
            i = 0
            for v in a:
                while i < len(b) and b[i] != v:
                    i += 1
                if i == len(b):
                    return False
                i += 1
            return True
    elif datatype == 'dict':
        if method == 'strict':
            return a == b
        else:
            return compare_dict_allow_more_present(a, b)
    elif datatype == 'set':
        set_a = set(a)
        set_b = set(b)
        if method == 'strict':
            return set_a == set_b
        else:
            return set_b >= set_a
    elif datatype == 'set(dict)':
        for av in a:
            found = False
            for bv in b:
                if compare_dict_allow_more_present(av, bv):
                    found = True
                    break
            if not found:
                return False
        if method == 'strict':
            for bv in b:
                found = False
                for av in a:
                    if compare_dict_allow_more_present(av, bv):
                        found = True
                        break
                if not found:
                    return False
        return True