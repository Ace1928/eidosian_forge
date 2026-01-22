from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def generic_hash(obj):
    """
    Compute a hash of all the fields in the object
    :param obj: Object whose hash needs to be computed
    :return: a hash value for the object
    """
    sum = 0
    for field in obj.attribute_map.keys():
        field_value = getattr(obj, field)
        if isinstance(field_value, list):
            for value in field_value:
                sum = sum + hash(value)
        elif isinstance(field_value, dict):
            for k, v in field_value.items():
                sum = sum + hash(hash(k) + hash(':') + hash(v))
        else:
            sum = sum + hash(getattr(obj, field))
    return sum