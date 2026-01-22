from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_assign_forward(self):
    """config assign forward command"""
    if self.nvo3_gw_enhanced:
        cmd = 'assign forward nvo3-gateway enhanced %s' % self.nvo3_gw_enhanced
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)
    if self.nvo3_prevent_loops:
        cmd = 'assign forward nvo3 f-linecard compatibility enable'
        exist = is_config_exist(self.config, cmd)
        if self.nvo3_prevent_loops == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)
    if self.nvo3_acl_extend:
        cmd = 'assign forward nvo3 acl extend enable'
        exist = is_config_exist(self.config, cmd)
        if self.nvo3_acl_extend == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)
    if self.nvo3_service_extend:
        cmd = 'assign forward nvo3 service extend enable'
        exist = is_config_exist(self.config, cmd)
        if self.nvo3_service_extend == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)
    if self.nvo3_eth_trunk_hash:
        cmd = 'assign forward nvo3 eth-trunk hash enable'
        exist = is_config_exist(self.config, cmd)
        if self.nvo3_eth_trunk_hash == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)
    if self.nvo3_ecmp_hash:
        cmd = 'assign forward nvo3 ecmp hash enable'
        exist = is_config_exist(self.config, cmd)
        if self.nvo3_ecmp_hash == 'enable':
            if not exist:
                self.cli_add_command(cmd)
        elif exist:
            self.cli_add_command(cmd, undo=True)