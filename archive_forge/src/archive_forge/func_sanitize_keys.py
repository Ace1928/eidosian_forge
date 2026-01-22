from __future__ import absolute_import, division, print_function
import datetime
import os
from collections import deque
from itertools import chain
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.warnings import warn
from ansible.module_utils.errors import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.six import (
from ansible.module_utils.common.validation import (
def sanitize_keys(obj, no_log_strings, ignore_keys=frozenset()):
    """Sanitize the keys in a container object by removing ``no_log`` values from key names.

    This is a companion function to the :func:`remove_values` function. Similar to that function,
    we make use of ``deferred_removals`` to avoid hitting maximum recursion depth in cases of
    large data structures.

    :arg obj: The container object to sanitize. Non-container objects are returned unmodified.
    :arg no_log_strings: A set of string values we do not want logged.
    :kwarg ignore_keys: A set of string values of keys to not sanitize.

    :returns: An object with sanitized keys.
    """
    deferred_removals = deque()
    no_log_strings = [to_native(s, errors='surrogate_or_strict') for s in no_log_strings]
    new_value = _sanitize_keys_conditions(obj, no_log_strings, ignore_keys, deferred_removals)
    while deferred_removals:
        old_data, new_data = deferred_removals.popleft()
        if isinstance(new_data, Mapping):
            for old_key, old_elem in old_data.items():
                if old_key in ignore_keys or old_key.startswith('_ansible'):
                    new_data[old_key] = _sanitize_keys_conditions(old_elem, no_log_strings, ignore_keys, deferred_removals)
                else:
                    new_key = _remove_values_conditions(old_key, no_log_strings, None)
                    new_data[new_key] = _sanitize_keys_conditions(old_elem, no_log_strings, ignore_keys, deferred_removals)
        else:
            for elem in old_data:
                new_elem = _sanitize_keys_conditions(elem, no_log_strings, ignore_keys, deferred_removals)
                if isinstance(new_data, MutableSequence):
                    new_data.append(new_elem)
                elif isinstance(new_data, MutableSet):
                    new_data.add(new_elem)
                else:
                    raise TypeError('Unknown container type encountered when removing private values from keys')
    return new_value