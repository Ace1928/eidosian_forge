from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def del_all_commands(self, obj_in_have):
    commands = []
    if not obj_in_have:
        return commands
    for m in obj_in_have.get('members', []):
        commands.append('interface' + ' ' + m['member'])
        commands.append('no channel-group')
    return commands