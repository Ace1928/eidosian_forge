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
def get_nested_config(proposed_child, existing_children):
    """
        This method is used for stiping off the outer layers of the child dictionaries so only the configuration
        key, value pairs are returned.
        :param proposed_child: Type dict.
                               The dictionary that represents the child config.
        :param existing_children: Type list.
                                  The list of existing child config dictionaries.
        :return: The child's class as str (root config dict key), the child's proposed config dict, and the child's
                 existing configuration dict.
        """
    for key in proposed_child.keys():
        child_class = key
        proposed_config = proposed_child[key]['attributes']
        existing_config = None
        for child in existing_children:
            if child.get(child_class):
                existing_config = child[key]['attributes']
                if set(proposed_config.items()).issubset(set(existing_config.items())):
                    break
                existing_config = None
    return (child_class, proposed_config, existing_config)