from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_replaced_delete_list(self, commands, have):
    matched = []
    for cmd in commands:
        name = cmd.get('name', None)
        interface_name = name.replace('/', '%2f')
        mapping_list = cmd.get('mapping', [])
        matched_interface_name = None
        matched_mapping_list = []
        for existing in have:
            have_name = existing.get('name', None)
            have_interface_name = have_name.replace('/', '%2f')
            have_mapping_list = existing.get('mapping', [])
            if interface_name == have_interface_name:
                matched_interface_name = have_interface_name
                matched_mapping_list = have_mapping_list
        if mapping_list and matched_mapping_list:
            returned_mapping_list = []
            for mapping in mapping_list:
                service_vlan = mapping.get('service_vlan', None)
                for matched_mapping in matched_mapping_list:
                    matched_service_vlan = matched_mapping.get('service_vlan', None)
                    if matched_service_vlan and service_vlan:
                        if matched_service_vlan == service_vlan:
                            priority = mapping.get('priority', None)
                            have_priority = matched_mapping.get('priority', None)
                            inner_vlan = mapping.get('inner_vlan', None)
                            have_inner_vlan = matched_mapping.get('inner_vlan', None)
                            dot1q_tunnel = mapping.get('dot1q_tunnel', False)
                            have_dot1q_tunnel = matched_mapping.get('dot1q_tunnel', False)
                            vlan_ids = mapping.get('vlan_ids', [])
                            have_vlan_ids = matched_mapping.get('vlan_ids', [])
                            if priority != have_priority:
                                returned_mapping_list.append(mapping)
                            elif inner_vlan != have_inner_vlan:
                                returned_mapping_list.append(mapping)
                            elif dot1q_tunnel != have_dot1q_tunnel:
                                returned_mapping_list.append(mapping)
                            elif sorted(vlan_ids) != sorted(have_vlan_ids):
                                returned_mapping_list.append(mapping)
            if returned_mapping_list:
                matched.append({'name': interface_name, 'mapping': returned_mapping_list})
    return matched