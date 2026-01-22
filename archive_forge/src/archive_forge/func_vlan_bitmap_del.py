from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_bitmap_del(self, oldmap, delmap):
    """vlan del bitmap"""
    vlan_bit = ['0'] * 1024
    if not oldmap or len(oldmap) == 0:
        return ''.join(vlan_bit)
    if len(oldmap) != 1024 or len(delmap) != 1024:
        self.module.fail_json(msg='Error: vlan bitmap is invalid.')
    for num in range(1024):
        tmp = int(delmap[num], 16) & int(oldmap[num], 16)
        vlan_bit[num] = hex(tmp)[2]
    vlan_xml = ''.join(vlan_bit)
    return vlan_xml