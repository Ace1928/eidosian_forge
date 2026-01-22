from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
@staticmethod
def get_diff_child(child_class, proposed_child, existing_child):
    """
        This method is used to get the difference between a proposed and existing child configs. The get_nested_config()
        method should be used to return the proposed and existing config portions of child.
        :param child_class: Type str.
                            The root class (dict key) for the child dictionary.
        :param proposed_child: Type dict.
                               The config portion of the proposed child dictionary.
        :param existing_child: Type dict.
                               The config portion of the existing child dictionary.
        :return: The child config with only values that are updated. If the proposed dictionary has no updates to make
                 to what exists on the APIC, then None is returned.
        """
    update_config = {child_class: {'attributes': {}}}
    for key, value in proposed_child.items():
        existing_field = existing_child.get(key)
        if value != existing_field:
            update_config[child_class]['attributes'][key] = value
    if not update_config[child_class]['attributes']:
        return None
    return update_config