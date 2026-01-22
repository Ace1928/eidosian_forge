from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_peer(self):
    """configure evpn bgp peer command"""
    if self.as_number and self.peer_address:
        cmd = 'peer %s as-number %s' % (self.peer_address, self.as_number)
        exist = is_config_exist(self.config, cmd)
        if not exist:
            self.module.fail_json(msg='Error:  The peer session %s does not exist or the peer already exists in another as-number.' % self.peer_address)
        cmd = 'bgp %s' % self.bgp_instance
        self.cli_add_command(cmd)
        cmd = 'l2vpn-family evpn'
        self.cli_add_command(cmd)
        exist_l2vpn = is_config_exist(self.config, cmd)
        if self.peer_enable:
            cmd = 'peer %s enable' % self.peer_address
            if exist_l2vpn:
                exist = is_config_exist(self.config_list[1], cmd)
                if self.peer_enable == 'true' and (not exist):
                    self.cli_add_command(cmd)
                    self.changed = True
                elif self.peer_enable == 'false' and exist:
                    self.cli_add_command(cmd, undo=True)
                    self.changed = True
            else:
                self.cli_add_command(cmd)
                self.changed = True
        if self.advertise_router_type:
            cmd = 'peer %s advertise %s' % (self.peer_address, self.advertise_router_type)
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                self.cli_add_command(cmd)
                self.changed = True
            elif self.state == 'absent' and exist:
                self.cli_add_command(cmd, undo=True)
                self.changed = True
    elif self.peer_group_name:
        cmd_1 = 'group %s external' % self.peer_group_name
        exist_1 = is_config_exist(self.config, cmd_1)
        cmd_2 = 'group %s internal' % self.peer_group_name
        exist_2 = is_config_exist(self.config, cmd_2)
        exist = False
        if exist_1:
            exist = True
        if exist_2:
            exist = True
        if not exist:
            self.module.fail_json(msg='Error: The peer-group %s does not exist.' % self.peer_group_name)
        cmd = 'bgp %s' % self.bgp_instance
        self.cli_add_command(cmd)
        cmd = 'l2vpn-family evpn'
        self.cli_add_command(cmd)
        exist_l2vpn = is_config_exist(self.config, cmd)
        if self.peer_enable:
            cmd = 'peer %s enable' % self.peer_group_name
            if exist_l2vpn:
                exist = is_config_exist(self.config_list[1], cmd)
                if self.peer_enable == 'true' and (not exist):
                    self.cli_add_command(cmd)
                    self.changed = True
                elif self.peer_enable == 'false' and exist:
                    self.cli_add_command(cmd, undo=True)
                    self.changed = True
            else:
                self.cli_add_command(cmd)
                self.changed = True
        if self.advertise_router_type:
            cmd = 'peer %s advertise %s' % (self.peer_group_name, self.advertise_router_type)
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                self.cli_add_command(cmd)
                self.changed = True
            elif self.state == 'absent' and exist:
                self.cli_add_command(cmd, undo=True)
                self.changed = True