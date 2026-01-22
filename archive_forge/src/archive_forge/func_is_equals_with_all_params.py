from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def is_equals_with_all_params(payload, connection, version, api_call_object, is_access_rule):
    if is_access_rule and 'action' in payload:
        payload_for_show = extract_payload_with_some_params(payload, ['name', 'uid', 'layer'])
        code, response = send_request(connection, version, 'show-' + api_call_object, payload_for_show)
        exist_action = response['action']['name']
        if exist_action.lower() != payload['action'].lower():
            if payload['action'].lower() != 'Apply Layer'.lower() or exist_action.lower() != 'Inner Layer'.lower():
                return False
    if not is_equals_with_position_param(payload, connection, version, api_call_object):
        return False
    return True