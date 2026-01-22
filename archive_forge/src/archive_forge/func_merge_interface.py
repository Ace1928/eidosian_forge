from __future__ import (absolute_import, division, print_function)
import re
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def merge_interface(self, ifname, mtu):
    """ Merge interface mtu."""
    xmlstr = ''
    change = False
    command = 'interface %s' % ifname
    self.cli_add_command(command)
    if self.state == 'present':
        if mtu and self.intf_info['ifMtu'] != mtu:
            command = 'mtu %s' % mtu
            self.cli_add_command(command)
            self.updates_cmd.append('mtu %s' % mtu)
            change = True
    elif self.intf_info['ifMtu'] != '1500' and self.intf_info['ifMtu']:
        command = 'mtu 1500'
        self.cli_add_command(command)
        self.updates_cmd.append('undo mtu')
        change = True
    return