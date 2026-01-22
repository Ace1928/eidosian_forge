from __future__ import (absolute_import, division, print_function)
from ansible.module_utils import basic
@staticmethod
def get_not_none_values_from_dict(parameters, keys):
    return dict(((key, value) for key, value in parameters.items() if key in keys and value is not None))