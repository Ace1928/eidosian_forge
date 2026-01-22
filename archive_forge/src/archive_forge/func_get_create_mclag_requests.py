from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_create_mclag_requests(self, want, commands):
    requests = []
    path = 'data/openconfig-mclag:mclag/mclag-domains/mclag-domain'
    method = PATCH
    payload = self.build_create_payload(want, commands)
    if payload:
        request = {'path': path, 'method': method, 'data': payload}
        requests.append(request)
    if 'gateway_mac' in commands and commands['gateway_mac'] is not None:
        gateway_mac_path = 'data/openconfig-mclag:mclag/mclag-gateway-macs/mclag-gateway-mac'
        gateway_mac_method = PATCH
        gateway_mac_payload = {'openconfig-mclag:mclag-gateway-mac': [{'gateway-mac': commands['gateway_mac'], 'config': {'gateway-mac': commands['gateway_mac']}}]}
        request = {'path': gateway_mac_path, 'method': gateway_mac_method, 'data': gateway_mac_payload}
        requests.append(request)
    if 'unique_ip' in commands and commands['unique_ip'] is not None:
        if commands['unique_ip']['vlans'] and commands['unique_ip']['vlans'] is not None:
            unique_ip_path = 'data/openconfig-mclag:mclag/vlan-interfaces/vlan-interface'
            unique_ip_method = PATCH
            unique_ip_payload = self.build_create_unique_ip_payload(commands['unique_ip']['vlans'])
            request = {'path': unique_ip_path, 'method': unique_ip_method, 'data': unique_ip_payload}
            requests.append(request)
    if 'peer_gateway' in commands and commands['peer_gateway'] is not None:
        if commands['peer_gateway']['vlans'] and commands['peer_gateway']['vlans'] is not None:
            peer_gateway_path = 'data/openconfig-mclag:mclag/vlan-ifs/vlan-if'
            peer_gateway_method = PATCH
            peer_gateway_payload = self.build_create_peer_gateway_payload(commands['peer_gateway']['vlans'])
            request = {'path': peer_gateway_path, 'method': peer_gateway_method, 'data': peer_gateway_payload}
            requests.append(request)
    if 'members' in commands and commands['members'] is not None:
        if commands['members']['portchannels'] and commands['members']['portchannels'] is not None:
            portchannel_path = 'data/openconfig-mclag:mclag/interfaces/interface'
            portchannel_method = PATCH
            portchannel_payload = self.build_create_portchannel_payload(want, commands['members']['portchannels'])
            request = {'path': portchannel_path, 'method': portchannel_method, 'data': portchannel_payload}
            requests.append(request)
    return requests