from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def startup_next_pat_file(self):
    """set next patch file"""
    commands = list()
    cmd = {'output': None, 'command': ''}
    if self.slot:
        if self.slot == 'all':
            cmd['command'] = 'startup patch %s %s' % (self.patch_file, self.slot)
            commands.append(cmd)
            self.updates_cmd.append('startup patch %s %s' % (self.patch_file, self.slot))
            run_commands(self.module, commands)
            self.changed = True
        else:
            cmd['command'] = 'startup patch %s slot %s' % (self.patch_file, self.slot)
            commands.append(cmd)
            self.updates_cmd.append('startup patch %s slot %s' % (self.patch_file, self.slot))
            run_commands(self.module, commands)
            self.changed = True
    if not self.slot:
        cmd['command'] = 'startup patch %s' % self.patch_file
        commands.append(cmd)
        self.updates_cmd.append('startup patch %s' % self.patch_file)
        run_commands(self.module, commands)
        self.changed = True