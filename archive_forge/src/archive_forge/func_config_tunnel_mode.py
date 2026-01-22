from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_tunnel_mode(self):
    """config tunnel mode vxlan"""
    if self.tunnel_mode_vxlan:
        cmd = 'ip tunnel mode vxlan'
        exist = is_config_exist(self.config, cmd)
        if self.tunnel_mode_vxlan == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)