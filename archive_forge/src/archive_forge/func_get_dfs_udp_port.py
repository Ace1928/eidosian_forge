from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def get_dfs_udp_port(config):
    """get dfs udp port"""
    get = re.findall('udp port (\\d+)', config)
    if not get:
        return None
    else:
        return get[0]