from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def get_vbdif_vpn(config):
    """get ip vpn name of interface vbdif"""
    get = re.findall('ip binding vpn-instance (\\S+)', config)
    if not get:
        return None
    else:
        return get[0]