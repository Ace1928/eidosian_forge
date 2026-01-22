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
def tuplize(d):
    """
    This function takes a dictionary and converts it to a list of tuples recursively.
    :param d: A dictionary.
    :return: List of tuples.
    """
    list_of_tuples = []
    key_list = sorted(list(d.keys()))
    for key in key_list:
        if isinstance(d[key], list):
            if d[key] and isinstance(d[key][0], dict):
                sub_tuples = []
                for sub_dict in d[key]:
                    sub_tuples.append(tuplize(sub_dict))
                list_of_tuples.append((sub_tuples is None, key, sub_tuples))
            else:
                list_of_tuples.append((d[key] is None, key, d[key]))
        elif isinstance(d[key], dict):
            tupled_value = tuplize(d[key])
            list_of_tuples.append((tupled_value is None, key, tupled_value))
        else:
            list_of_tuples.append((d[key] is None, key, d[key]))
    return list_of_tuples