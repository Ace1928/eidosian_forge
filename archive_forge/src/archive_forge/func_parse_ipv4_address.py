from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_ipv4_address(self, value, intf_type='ethernet'):
    ipv4 = {}
    match = re.search('Internet Address is (.+)$', value, re.M)
    if match:
        address = match.group(1)
        addr = address.split('/')[0]
        ipv4['address'] = address.split('/')[0]
        ipv4['masklen'] = address.split('/')[1]
        self.facts['all_ipv4_addresses'].append(addr)
    return ipv4