from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def compare_nested_dict(modify_setting_payload, existing_setting_payload):
    """compare existing and requested setting values of identity pool in case of modify operations
    if both are same return True"""
    for key, val in modify_setting_payload.items():
        if existing_setting_payload is None or existing_setting_payload.get(key) is None:
            return False
        elif isinstance(val, dict):
            if not compare_nested_dict(val, existing_setting_payload.get(key)):
                return False
        elif val != existing_setting_payload.get(key):
            return False
    return True