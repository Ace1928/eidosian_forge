from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
def snake_dict_to_camel_dict(snake_dict, capitalize_first=False):
    """
    Perhaps unexpectedly, snake_dict_to_camel_dict returns dromedaryCase
    rather than true CamelCase. Passing capitalize_first=True returns
    CamelCase. The default remains False as that was the original implementation
    """

    def camelize(complex_type, capitalize_first=False):
        if complex_type is None:
            return
        new_type = type(complex_type)()
        if isinstance(complex_type, dict):
            for key in complex_type:
                new_type[_snake_to_camel(key, capitalize_first)] = camelize(complex_type[key], capitalize_first)
        elif isinstance(complex_type, list):
            for i in range(len(complex_type)):
                new_type.append(camelize(complex_type[i], capitalize_first))
        else:
            return complex_type
        return new_type
    return camelize(snake_dict, capitalize_first)