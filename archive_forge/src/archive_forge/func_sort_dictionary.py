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
def sort_dictionary(d):
    """
    This function sorts values of a dictionary recursively.
    :param d: A dictionary.
    :return: Dictionary with sorted elements.
    """
    sorted_d = {}
    for key in d:
        if isinstance(d[key], list):
            if d[key] and isinstance(d[key][0], dict):
                sorted_value = sort_list_of_dictionary(d[key])
                sorted_d[key] = sorted_value
            else:
                sorted_d[key] = sorted(d[key])
        elif isinstance(d[key], dict):
            sorted_d[key] = sort_dictionary(d[key])
        else:
            sorted_d[key] = d[key]
    return sorted_d