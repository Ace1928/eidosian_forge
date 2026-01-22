from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def generate_delete_commands(self, obj):
    """Generate CLI commands to remove non-default settings.
        obj: dict of attrs to remove
        """
    commands = []
    name = obj.get('name')
    if 'dot1q' in obj:
        commands.append('no encapsulation dot1q')
    if 'redirects' in obj:
        if not self.check_existing(name, 'has_secondary') or re.match('N[35679]', self.platform):
            commands.append('ip redirects')
    if 'ipv6_redirects' in obj:
        if not self.check_existing(name, 'has_secondary') or re.match('N[35679]', self.platform):
            commands.append('ipv6 redirects')
    if 'unreachables' in obj:
        commands.append('no ip unreachables')
    if 'ipv4' in obj:
        commands.append('no ip address')
    if 'ipv6' in obj:
        commands.append('no ipv6 address')
    if 'evpn_multisite_tracking' in obj:
        have = self.existing_facts.get(name, {})
        if have.get('evpn_multisite_tracking', False) is not False:
            cmd = 'no evpn multisite %s' % have.get('evpn_multisite_tracking')
            commands.append(cmd)
    return commands