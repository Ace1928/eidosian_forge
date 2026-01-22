from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_edge_position_in_section(connection, version, identifier, section_name, edge):
    code, response = send_request(connection, version, 'show-layer-structure', {'name': identifier, 'details-level': 'uid'})
    if 'code' in response and response['code'] == 'generic_err_command_not_found':
        raise ValueError('The use of the relative_position field with a section as its value is available only for version 1.7.1 with JHF take 42 and above')
    sections_in_layer = response['root-section']['children']
    for section in sections_in_layer:
        if section['name'] == section_name:
            return int(section[edge + '-rule'])
    return None