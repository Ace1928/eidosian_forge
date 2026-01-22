from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config, ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def undo_trap_source_interface(self):
    """ Undo source interface for trap """
    cmd = 'undo snmp-agent trap source'
    self.updates_cmd.append(cmd)
    cmds = list()
    cmds.append(cmd)
    self.cli_load_config(cmds)
    self.changed = True