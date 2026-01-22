from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.security_zones.security_zones import (
def parse_host_inbound_traffic(self, host_inbound_traffic):
    temp_hit = {}
    if 'protocols' in host_inbound_traffic:
        temp_hit['protocols'] = []
        if isinstance(host_inbound_traffic['protocols'], dict):
            host_inbound_traffic['protocols'] = [host_inbound_traffic['protocols']]
        for protocol in host_inbound_traffic['protocols']:
            temp_protocol = {}
            temp_protocol['name'] = protocol['name']
            if 'except' in protocol:
                temp_protocol['except'] = True
            temp_hit['protocols'].append(temp_protocol)
    if 'system-services' in host_inbound_traffic:
        temp_hit['system_services'] = []
        if isinstance(host_inbound_traffic['system-services'], dict):
            host_inbound_traffic['system-services'] = [host_inbound_traffic['system-services']]
        for system_services in host_inbound_traffic['system-services']:
            temp_system_services = {}
            temp_system_services['name'] = system_services['name']
            if 'except' in system_services:
                temp_system_services['except'] = True
            temp_hit['system_services'].append(temp_system_services)
    return temp_hit