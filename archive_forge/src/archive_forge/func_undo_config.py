from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def undo_config(self):
    """ Undo configure by cli """
    cmd = 'undo snmp-agent sys-info contact'
    self.updates_cmd.append(cmd)
    cmds = list()
    cmds.append(cmd)
    self.cli_load_config(cmds)
    self.changed = True