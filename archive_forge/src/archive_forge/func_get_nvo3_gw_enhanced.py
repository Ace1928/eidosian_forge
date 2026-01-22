from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def get_nvo3_gw_enhanced(cmp_cfg):
    """get the Layer 3 VXLAN Gateway to Work in Non-loopback Mode """
    get = re.findall('assign forward nvo3-gateway enhanced (l[2|3])', cmp_cfg)
    if not get:
        return None
    else:
        return get[0]