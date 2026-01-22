from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_region(self, vlanid_list):
    """ convert vlan range to vlan list """
    vlan_list = []
    peerlistlen = len(vlanid_list)
    for num in range(peerlistlen):
        if vlanid_list[num].isdigit():
            vlan_list.append(vlanid_list[num])
        else:
            vlan_s = self.vlan_series(vlanid_list[num].split('-'))
            vlan_list.extend(vlan_s)
    return vlan_list