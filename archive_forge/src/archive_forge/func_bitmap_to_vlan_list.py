from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def bitmap_to_vlan_list(self, bitmap):
    """convert VLAN bitmap to VLAN list"""
    vlan_list = list()
    if not bitmap:
        return vlan_list
    for i in range(len(bitmap)):
        if bitmap[i] == '0':
            continue
        bit = int(bitmap[i], 16)
        if bit & 8:
            vlan_list.append(str(i * 4))
        if bit & 4:
            vlan_list.append(str(i * 4 + 1))
        if bit & 2:
            vlan_list.append(str(i * 4 + 2))
        if bit & 1:
            vlan_list.append(str(i * 4 + 3))
    return vlan_list