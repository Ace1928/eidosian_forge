from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def isvalidlsaholdarrivalinterval(self):
    """check whether the input ospf lsa hold arrival interval is valid"""
    if not self.lsaaholdinterval.isdigit():
        return False
    if int(self.lsaaholdinterval) > 5000 or int(self.lsaaholdinterval) < 0:
        return False
    return True