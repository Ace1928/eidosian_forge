from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def compare_data(self):
    """compare new data and old data"""
    state = self.state
    change = False
    if state == 'present':
        if self.igmp_info_data['igmp_info']:
            for data in self.igmp_info_data['igmp_info']:
                if self.addr_family == data['addrFamily'] and str(self.vlan_id) == data['vlanId']:
                    if self.igmp:
                        if self.igmp != data['snoopingEnable']:
                            change = True
                    if self.version:
                        if str(self.version) != data['version']:
                            change = True
                    if self.proxy:
                        if self.proxy != data['proxyEnable']:
                            change = True
        else:
            change = True
    elif self.igmp_info_data['igmp_info']:
        change = True
    return change