from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def remove_vrf(host, udp, proposed, existing):
    commands = []
    if existing.get('vrf'):
        commands.append('no snmp-server host {0} use-vrf                     {1} udp-port {2}'.format(host, proposed.get('vrf'), udp))
    return commands