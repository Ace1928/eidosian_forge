from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_bitmap_undo(bitmap):
    """convert vlan bitmap to undo bitmap"""
    vlan_bit = ['F'] * 1024
    if not bitmap or len(bitmap) == 0:
        return ''.join(vlan_bit)
    bit_len = len(bitmap)
    for num in range(bit_len):
        undo = ~int(bitmap[num], 16) & 15
        vlan_bit[num] = hex(undo)[2]
    return ''.join(vlan_bit)