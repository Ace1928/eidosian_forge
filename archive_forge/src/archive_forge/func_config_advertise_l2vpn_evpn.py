from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_advertise_l2vpn_evpn(self):
    """configure advertise l2vpn evpn"""
    cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
    exist = is_config_exist(self.config, cmd)
    if not exist:
        self.module.fail_json(msg='Error: The VPN instance name %s does not exist.' % self.vpn_name)
    config_vpn_list = self.config.split(cmd)
    cmd = 'ipv4-family vpn-instance'
    exist_vpn = is_config_exist(config_vpn_list[1], cmd)
    cmd_l2vpn = 'advertise l2vpn evpn'
    if exist_vpn:
        config_vpn = config_vpn_list[1].split('ipv4-family vpn-instance')
        exist_l2vpn = is_config_exist(config_vpn[0], cmd_l2vpn)
    else:
        exist_l2vpn = is_config_exist(config_vpn_list[1], cmd_l2vpn)
    cmd = 'advertise l2vpn evpn'
    if self.advertise_l2vpn_evpn == 'enable' and (not exist_l2vpn):
        cmd = 'bgp %s' % self.bgp_instance
        self.cli_add_command(cmd)
        cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
        self.cli_add_command(cmd)
        cmd = 'advertise l2vpn evpn'
        self.cli_add_command(cmd)
        self.changed = True
    elif self.advertise_l2vpn_evpn == 'disable' and exist_l2vpn:
        cmd = 'bgp %s' % self.bgp_instance
        self.cli_add_command(cmd)
        cmd = 'ipv4-family vpn-instance %s' % self.vpn_name
        self.cli_add_command(cmd)
        cmd = 'advertise l2vpn evpn'
        self.cli_add_command(cmd, undo=True)
        self.changed = True