from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_l2_sub_intf_valid(self, ifname):
    """check l2 sub interface valid"""
    if ifname.count('.') != 1:
        return False
    if_num = ifname.split('.')[1]
    if not if_num.isdigit():
        return False
    if int(if_num) < 1 or int(if_num) > 4096:
        self.module.fail_json(msg='Error: Sub-interface number is not in the range from 1 to 4096.')
        return False
    if not get_interface_type(ifname):
        return False
    return True