from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
def del_all(self, diff):
    commands = []
    base = 'no lacp system-'
    diff = diff.get('system')
    if diff:
        if 'priority' in diff:
            commands.append(base + 'priority')
        if 'mac' in diff:
            commands.append(base + 'mac')
    return commands