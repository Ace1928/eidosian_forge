from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vlan_mapping_requests(self, commands, have, is_delete_all):
    """ Get list of requests to delete vlan mapping configurations
        for all interfaces specified by the commands
        """
    url = 'data/openconfig-interfaces:interfaces/interface={}/openconfig-interfaces-ext:mapped-vlans/mapped-vlan={}'
    priority_url = '/ingress-mapping/config/mapped-vlan-priority'
    vlan_ids_url = '/match/single-tagged/config/vlan-ids={}'
    method = 'DELETE'
    requests = []
    if is_delete_all:
        for cmd in commands:
            name = cmd.get('name', None)
            interface_name = name.replace('/', '%2f')
            mapping_list = cmd.get('mapping', [])
            if mapping_list:
                for mapping in mapping_list:
                    service_vlan = mapping.get('service_vlan', None)
                    path = url.format(interface_name, service_vlan)
                    request = {'path': path, 'method': method}
                    requests.append(request)
        return requests
    else:
        for cmd in commands:
            name = cmd.get('name', None)
            interface_name = name.replace('/', '%2f')
            mapping_list = cmd.get('mapping', [])
            have_interface_name = None
            have_mapping_list = []
            for tmp in have:
                tmp_name = tmp.get('name', None)
                tmp_interface_name = tmp_name.replace('/', '%2f')
                tmp_mapping_list = tmp.get('mapping', [])
                if interface_name == tmp_interface_name:
                    have_interface_name = tmp_interface_name
                    have_mapping_list = tmp_mapping_list
            if mapping_list:
                for mapping in mapping_list:
                    service_vlan = mapping.get('service_vlan', None)
                    vlan_ids = mapping.get('vlan_ids', None)
                    priority = mapping.get('priority', None)
                    have_service_vlan = None
                    have_vlan_ids = None
                    have_priority = None
                    for have_mapping in have_mapping_list:
                        if have_mapping.get('service_vlan', None) == service_vlan:
                            have_service_vlan = have_mapping.get('service_vlan', None)
                            have_vlan_ids = have_mapping.get('vlan_ids', None)
                            have_priority = have_mapping.get('priority', None)
                    if service_vlan and have_service_vlan:
                        if vlan_ids or priority:
                            if priority and have_priority:
                                path = url.format(interface_name, service_vlan) + priority_url
                                request = {'path': path, 'method': method}
                                requests.append(request)
                            if vlan_ids and have_vlan_ids:
                                vlan_ids_str = ''
                                same_vlan_ids_list = self.get_vlan_ids_diff(vlan_ids, have_vlan_ids, same=True)
                                if same_vlan_ids_list:
                                    for vlan in same_vlan_ids_list:
                                        if vlan_ids_str:
                                            vlan_ids_str = vlan_ids_str + '%2C' + vlan.replace('-', '..')
                                        else:
                                            vlan_ids_str = vlan.replace('-', '..')
                                    path = url.format(interface_name, service_vlan) + vlan_ids_url.format(vlan_ids_str)
                                    request = {'path': path, 'method': method}
                                    requests.append(request)
                        else:
                            path = url.format(interface_name, service_vlan)
                            request = {'path': path, 'method': method}
                            requests.append(request)
            elif have_mapping_list:
                for mapping in have_mapping_list:
                    service_vlan = mapping.get('service_vlan', None)
                    path = url.format(interface_name, service_vlan)
                    request = {'path': path, 'method': method}
                    requests.append(request)
        return requests