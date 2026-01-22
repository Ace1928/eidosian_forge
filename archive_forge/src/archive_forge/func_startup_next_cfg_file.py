from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def startup_next_cfg_file(self):
    """set next cfg file"""
    commands = list()
    cmd = {'output': None, 'command': ''}
    if self.slot:
        cmd['command'] = 'startup saved-configuration %s slot %s' % (self.cfg_file, self.slot)
        commands.append(cmd)
        self.updates_cmd.append('startup saved-configuration %s slot %s' % (self.cfg_file, self.slot))
        run_commands(self.module, commands)
        self.changed = True
    else:
        cmd['command'] = 'startup saved-configuration %s' % self.cfg_file
        commands.append(cmd)
        self.updates_cmd.append('startup saved-configuration %s' % self.cfg_file)
        run_commands(self.module, commands)
        self.changed = True