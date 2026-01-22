from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_vll_endpoints(self, data):
    facts = list()
    regex = 'End-point[0-9 ]*: +(?P<tagged>tagged|untagged) +(vlan +(?P<vlan>[0-9]+) +)?(inner- vlan +(?P<innervlan>[0-9]+) +)?(?P<port>e [0-9/]+|--)'
    matches = re.finditer(regex, data, re.IGNORECASE | re.DOTALL)
    for match in matches:
        f = match.groupdict()
        f['type'] = 'local'
        facts.append(f)
    regex = 'Vll-Peer +: +(?P<vllpeer>[0-9\\.]+).*Tunnel LSP +: +(?P<lsp>\\S+)'
    matches = re.finditer(regex, data, re.IGNORECASE | re.DOTALL)
    for match in matches:
        f = match.groupdict()
        f['type'] = 'remote'
        facts.append(f)
    return facts