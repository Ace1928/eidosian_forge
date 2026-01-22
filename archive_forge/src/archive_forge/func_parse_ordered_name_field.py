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
def parse_ordered_name_field(input_list, name_field_name):
    """Take a dict with key: value or key: list_of_values mappings and return a list of tuples"""
    result = []
    for index, entry in enumerate(input_list):
        if len(entry) != 1:
            raise ValueError('Entry #{index} in {name} must be a dictionary with exactly one key-value pair'.format(name=name_field_name, index=index + 1))
        try:
            result.extend(parse_name_field(entry, name_field_name=name_field_name))
        except (TypeError, ValueError) as exc:
            raise ValueError('Error while processing entry #{index} in {name}: {error}'.format(name=name_field_name, index=index + 1, error=exc))
    return result