from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_change_state(self):
    """set change state"""
    state = self.state
    change = False
    if self.features == 'vlan':
        self.get_igmp_vlan()
        change = self.compare_data()
    else:
        self.get_igmp_global()
        if state == 'present':
            if not self.igmp_info_data['igmp_info']:
                change = True
        elif self.igmp_info_data['igmp_info']:
            change = True
    self.changed = change