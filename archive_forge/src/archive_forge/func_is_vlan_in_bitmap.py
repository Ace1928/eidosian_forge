from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vlan_in_bitmap(vid, bitmap):
    """check is VLAN id in bitmap"""
    if is_vlan_bitmap_empty(bitmap):
        return False
    i = int(vid) // 4
    if i > len(bitmap):
        return False
    if int(bitmap[i]) & 8 >> int(vid) % 4:
        return True
    return False