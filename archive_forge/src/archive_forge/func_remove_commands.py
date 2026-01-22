from __future__ import absolute_import, division, print_function
import itertools
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def remove_commands(want, interface):
    commands = []
    if 'access_groups' not in want.keys() or not want['access_groups']:
        return commands
    for w in want['access_groups']:
        a_cmd = 'access-group'
        afi = 'ip' if w['afi'] == 'ipv4' else w['afi']
        if 'acls' in w.keys():
            for acl in w['acls']:
                commands.append('no ' + afi + ' ' + a_cmd + ' ' + acl['name'] + ' ' + acl['direction'])
    return commands