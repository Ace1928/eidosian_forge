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
def check_if_user_value_matches_resources_attr(attribute_name, resources_value_for_attr, user_provided_value_for_attr, exclude_attributes, default_attribute_values, res):
    if isinstance(default_attribute_values.get(attribute_name), dict):
        default_attribute_values = default_attribute_values.get(attribute_name)
    if isinstance(exclude_attributes.get(attribute_name), dict):
        exclude_attributes = exclude_attributes.get(attribute_name)
    if isinstance(resources_value_for_attr, list) or isinstance(user_provided_value_for_attr, list):
        if exclude_attributes.get(attribute_name):
            return
        if user_provided_value_for_attr is None and default_attribute_values.get(attribute_name) is not None:
            user_provided_value_for_attr = default_attribute_values.get(attribute_name)
        if resources_value_for_attr is None and user_provided_value_for_attr is None:
            return
        if resources_value_for_attr is None or user_provided_value_for_attr is None:
            res[0] = False
            return
        if resources_value_for_attr is not None and user_provided_value_for_attr is not None and (len(resources_value_for_attr) != len(user_provided_value_for_attr)):
            res[0] = False
            return
        if user_provided_value_for_attr and isinstance(user_provided_value_for_attr[0], dict):
            sorted_user_provided_value_for_attr = sort_list_of_dictionary(user_provided_value_for_attr)
            sorted_resources_value_for_attr = sort_list_of_dictionary(resources_value_for_attr)
        else:
            sorted_user_provided_value_for_attr = sorted(user_provided_value_for_attr)
            sorted_resources_value_for_attr = sorted(resources_value_for_attr)
        for index, resources_value_for_attr_part in enumerate(sorted_resources_value_for_attr):
            check_if_user_value_matches_resources_attr(attribute_name, resources_value_for_attr_part, sorted_user_provided_value_for_attr[index], exclude_attributes, default_attribute_values, res)
    elif isinstance(resources_value_for_attr, dict):
        if not resources_value_for_attr and user_provided_value_for_attr:
            res[0] = False
        for key in resources_value_for_attr:
            if user_provided_value_for_attr is not None and user_provided_value_for_attr:
                check_if_user_value_matches_resources_attr(key, resources_value_for_attr.get(key), user_provided_value_for_attr.get(key), exclude_attributes, default_attribute_values, res)
            elif exclude_attributes.get(key) is None:
                if default_attribute_values.get(key) is not None:
                    user_provided_value_for_attr = default_attribute_values.get(key)
                    check_if_user_value_matches_resources_attr(key, resources_value_for_attr.get(key), user_provided_value_for_attr, exclude_attributes, default_attribute_values, res)
                else:
                    res[0] = is_attr_assigned_default(default_attribute_values, attribute_name, resources_value_for_attr.get(key))
    elif resources_value_for_attr != user_provided_value_for_attr:
        if exclude_attributes.get(attribute_name) is None and default_attribute_values.get(attribute_name) is not None:
            if not is_attr_assigned_default(default_attribute_values, attribute_name, resources_value_for_attr):
                res[0] = False
        elif user_provided_value_for_attr is not None:
            res[0] = False