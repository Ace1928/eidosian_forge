from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
def is_valid_timedelta(value):
    if value == timedelta(10675199, 10085, 477581):
        return None
    return value