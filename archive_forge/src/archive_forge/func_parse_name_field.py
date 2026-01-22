from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def parse_name_field(input_dict, name_field_name=None):
    """Take a dict with key: value or key: list_of_values mappings and return a list of tuples"""
    error_str = '{key}' if name_field_name is None else '{key} in {name}'
    result = []
    for key, value in input_dict.items():
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, six.string_types):
                    raise TypeError(('Values %s must be strings' % error_str).format(key=key, name=name_field_name))
                if not entry:
                    raise ValueError(('Values for %s must not be empty strings' % error_str).format(key=key))
                result.append((key, entry))
        elif isinstance(value, six.string_types):
            if not value:
                raise ValueError(('Value for %s must not be an empty string' % error_str).format(key=key))
            result.append((key, value))
        else:
            raise TypeError(('Value for %s must be either a string or a list of strings' % error_str).format(key=key))
    return result