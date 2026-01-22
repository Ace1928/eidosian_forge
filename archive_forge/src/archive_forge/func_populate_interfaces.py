from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def populate_interfaces(self, interfaces):
    facts = dict()
    counters = {'description': 'Description: (.+)', 'macaddress': 'HWaddr: (\\S+)', 'type': 'Type: (\\S+)', 'vrf': 'vrf: (\\S+)', 'mtu': 'mtu (\\d+)', 'bandwidth': 'bandwidth (\\d+)', 'lineprotocol': 'line protocol is (\\S+)', 'operstatus': '^(?:.+) is (.+),'}
    for key, value in iteritems(interfaces):
        intf = dict()
        for fact, pattern in iteritems(counters):
            intf[fact] = self.parse_facts(pattern, value)
        facts[key] = intf
    return facts