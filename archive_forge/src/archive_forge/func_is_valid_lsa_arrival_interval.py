from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_valid_lsa_arrival_interval(self):
    """check whether the input ospf lsa arrival interval is valid"""
    if self.lsaainterval is None:
        return False
    if not self.lsaainterval.isdigit():
        return False
    if int(self.lsaainterval) > 10000 or int(self.lsaainterval) < 0:
        return False
    return True