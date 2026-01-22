from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vlan_bitmap_empty(bitmap):
    """check vlan bitmap empty"""
    if not bitmap or len(bitmap) == 0:
        return True
    bit_len = len(bitmap)
    for num in range(bit_len):
        if bitmap[num] != '0':
            return False
    return True