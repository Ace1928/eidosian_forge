from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def undo_config_ntp_auth_keyid(self):
    """Undo ntp authentication key-id"""
    commands = list()
    config_cli = 'undo ntp authentication-keyid %s' % self.key_id
    commands.append(config_cli)
    self.cli_load_config(commands)