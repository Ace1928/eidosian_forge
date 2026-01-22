from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def config_bridge_domain(self):
    """manage bridge domain configuration"""
    if not self.bridge_domain_id:
        return
    cmd = 'bridge-domain %s' % self.bridge_domain_id
    if not is_config_exist(self.config, cmd):
        self.module.fail_json(msg='Error: Bridge domain %s is not exist.' % self.bridge_domain_id)
    cmd = 'arp broadcast-suppress enable'
    exist = is_config_exist(self.config, cmd)
    if self.arp_suppress == 'enable' and (not exist):
        self.cli_add_command('bridge-domain %s' % self.bridge_domain_id)
        self.cli_add_command(cmd)
        self.cli_add_command('quit')
    elif self.arp_suppress == 'disable' and exist:
        self.cli_add_command('bridge-domain %s' % self.bridge_domain_id)
        self.cli_add_command(cmd, undo=True)
        self.cli_add_command('quit')