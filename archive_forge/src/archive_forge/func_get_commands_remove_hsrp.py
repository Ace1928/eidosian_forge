from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_remove_hsrp(group, interface):
    commands = ['interface {0}'.format(interface), 'no hsrp {0}'.format(group)]
    return commands