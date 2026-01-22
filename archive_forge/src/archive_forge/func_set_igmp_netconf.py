from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_igmp_netconf(self):
    """config netconf"""
    if self.features == 'vlan':
        self.set_vlanview_igmp()
    else:
        self.set_sysview_igmp()