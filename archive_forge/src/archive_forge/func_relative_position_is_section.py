from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def relative_position_is_section(connection, version, api_call_object, layer_or_package_payload, relative_position):
    if 'top' in relative_position or 'bottom' in relative_position:
        return True
    show_section_command = 'show-access-section' if 'access' in api_call_object else 'show-nat-section'
    relative_position_value = list(relative_position.values())[0]
    payload = {'name': relative_position_value}
    payload.update(layer_or_package_payload)
    code, response = send_request(connection, version, show_section_command, payload)
    if code == 200:
        return True
    return False