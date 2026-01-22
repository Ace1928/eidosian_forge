from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_create_vlan_mapping_request(self, interface_name, mapping):
    url = 'data/openconfig-interfaces:interfaces/interface={}/openconfig-interfaces-ext:mapped-vlans'
    body = {}
    method = 'PATCH'
    match_data = None
    service_vlan = mapping.get('service_vlan', None)
    priority = mapping.get('priority', None)
    vlan_ids = mapping.get('vlan_ids', [])
    dot1q_tunnel = mapping.get('dot1q_tunnel', None)
    inner_vlan = mapping.get('inner_vlan', None)
    if not dot1q_tunnel:
        if len(vlan_ids) > 1:
            raise Exception('When dot1q-tunnel is false only one VLAN ID can be passed to the vlan_ids list')
        if not vlan_ids and priority:
            match_data = None
        elif vlan_ids:
            if inner_vlan:
                match_data = {'double-tagged': {'config': {'inner-vlan-id': inner_vlan, 'outer-vlan-id': int(vlan_ids[0])}}}
            else:
                match_data = {'single-tagged': {'config': {'vlan-ids': [int(vlan_ids[0])]}}}
        if priority:
            ing_data = {'config': {'vlan-stack-action': 'SWAP', 'mapped-vlan-priority': priority}}
            egr_data = {'config': {'vlan-stack-action': 'SWAP', 'mapped-vlan-priority': priority}}
        else:
            ing_data = {'config': {'vlan-stack-action': 'SWAP'}}
            egr_data = {'config': {'vlan-stack-action': 'SWAP'}}
    else:
        if inner_vlan:
            raise Exception('Inner vlan can only be passed when dot1q_tunnel is false')
        if not vlan_ids and priority:
            match_data = None
        elif vlan_ids:
            vlan_ids_list = []
            for vlan in vlan_ids:
                vlan_ids_list.append(int(vlan))
            match_data = {'single-tagged': {'config': {'vlan-ids': vlan_ids_list}}}
        if priority:
            ing_data = {'config': {'vlan-stack-action': 'PUSH', 'mapped-vlan-priority': priority}}
            egr_data = {'config': {'vlan-stack-action': 'POP', 'mapped-vlan-priority': priority}}
        else:
            ing_data = {'config': {'vlan-stack-action': 'PUSH'}}
            egr_data = {'config': {'vlan-stack-action': 'POP'}}
    if match_data:
        body = {'openconfig-interfaces-ext:mapped-vlans': {'mapped-vlan': [{'vlan-id': service_vlan, 'config': {'vlan-id': service_vlan}, 'match': match_data, 'ingress-mapping': ing_data, 'egress-mapping': egr_data}]}}
    else:
        body = {'openconfig-interfaces-ext:mapped-vlans': {'mapped-vlan': [{'vlan-id': service_vlan, 'config': {'vlan-id': service_vlan}, 'ingress-mapping': ing_data, 'egress-mapping': egr_data}]}}
    request = {'path': url.format(interface_name), 'method': method, 'data': body}
    return request