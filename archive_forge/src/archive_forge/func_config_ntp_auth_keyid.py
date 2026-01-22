from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def config_ntp_auth_keyid(self):
    """Config ntp authentication keyid"""
    commands = list()
    if self.auth_type == 'encrypt':
        config_cli = 'ntp authentication-keyid %s authentication-mode %s cipher %s' % (self.key_id, self.auth_mode, self.password)
    else:
        config_cli = 'ntp authentication-keyid %s authentication-mode %s %s' % (self.key_id, self.auth_mode, self.password)
    commands.append(config_cli)
    if self.trusted_key != self.cur_trusted_key:
        if self.trusted_key == 'enable':
            config_cli_trust = 'ntp trusted authentication-keyid %s' % self.key_id
            commands.append(config_cli_trust)
        else:
            config_cli_trust = 'undo ntp trusted authentication-keyid %s' % self.key_id
            commands.append(config_cli_trust)
    self.cli_load_config(commands)