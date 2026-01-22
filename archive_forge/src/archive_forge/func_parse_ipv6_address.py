from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_ipv6_address(self, value):
    ipv6 = {}
    match_addr = re.search('IPv6 address:\\s*(\\S+)', value, re.M)
    if match_addr:
        addr = match_addr.group(1)
        ipv6['address'] = addr
        self.facts['all_ipv6_addresses'].append(addr)
    match_subnet = re.search('IPv6 subnet:\\s*(\\S+)', value, re.M)
    if match_subnet:
        ipv6['subnet'] = match_subnet.group(1)
    return ipv6