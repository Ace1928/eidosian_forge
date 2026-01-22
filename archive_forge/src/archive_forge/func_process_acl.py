from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def process_acl(self, acls, ip, deleted=False):
    commands = []
    no = ''
    if deleted:
        no = 'no '
    for acl in acls:
        port = ''
        if acl.get('port'):
            port = ' port'
        ag = ' access-group '
        if ip == 'ipv6':
            ag = ' traffic-filter '
        commands.append(no + ip + port + ag + acl['name'] + ' ' + acl['direction'])
    return commands