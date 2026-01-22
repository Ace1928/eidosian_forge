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
def update_class_type_attr_difference(update_class_details, existing_instance, attr_name, attr_class, input_attr_value):
    """
    Checks the difference and updates an attribute which is represented by a class
    instance. Not applicable if the attribute type is a primitive value.
    For example, if a class name is A with an attribute x, then if A.x = X(), then only
    this method works.
    :param update_class_details The instance which should be updated if there is change in
     attribute value
    :param existing_instance The instance  whose attribute value is compared with input
     attribute value
    :param attr_name Name of the attribute whose value should be compared
    :param attr_class Class type of the attribute
    :param input_attr_value The value of input attribute which should replaced the current
     value in case of mismatch
    :return: A boolean value indicating whether attribute value has been replaced
    """
    changed = False
    existing_attr_value = get_hashed_object(attr_class, getattr(existing_instance, attr_name))
    if input_attr_value is None:
        update_class_details.__setattr__(attr_name, existing_attr_value)
    else:
        changed = not input_attr_value.__eq__(existing_attr_value)
        if changed:
            update_class_details.__setattr__(attr_name, input_attr_value)
        else:
            update_class_details.__setattr__(attr_name, existing_attr_value)
    return changed