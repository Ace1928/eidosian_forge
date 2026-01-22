from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
def parse_filters(filters):
    """
    Parse get_option('filter') and return normalized version to be fed into filter_host().
    """
    result = []
    if filters is None:
        return result
    for index, filter in enumerate(filters):
        if not isinstance(filter, Mapping):
            raise AnsibleError('filter[{index}] must be a dictionary'.format(index=index + 1))
        if len(filter) != 1:
            raise AnsibleError('filter[{index}] must have exactly one key-value pair'.format(index=index + 1))
        key, value = list(filter.items())[0]
        if key not in _ALLOWED_KEYS:
            raise AnsibleError('filter[{index}] must have a {allowed_keys} key, not "{key}"'.format(index=index + 1, key=key, allowed_keys=' or '.join(('"{key}"'.format(key=key) for key in _ALLOWED_KEYS))))
        if not isinstance(value, (string_types, bool)):
            raise AnsibleError('filter[{index}].{key} must be a string, not {value_type}'.format(index=index + 1, key=key, value_type=type(value)))
        result.append(filter)
    return result