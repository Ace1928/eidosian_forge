from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def remove_src(host, udp, proposed, existing):
    commands = []
    if existing.get('src_intf'):
        commands.append('no snmp-server host {0} source-interface                     {1} udp-port {2}'.format(host, proposed.get('src_intf'), udp))
    return commands