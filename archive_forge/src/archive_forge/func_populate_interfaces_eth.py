from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.voss.voss import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_interfaces_eth(self, interfaces):
    facts = dict()
    for key, value in iteritems(interfaces):
        intf = dict()
        match = re.match('^\\d+\\s+(\\S+)\\s+\\w+\\s+\\w+\\s+(\\d+)\\s+([a-f\\d:]+)\\s+(\\w+)\\s+(\\w+)$', value)
        if match:
            intf['mediatype'] = match.group(1)
            intf['mtu'] = match.group(2)
            intf['macaddress'] = match.group(3)
            intf['adminstatus'] = match.group(4)
            intf['operstatus'] = match.group(5)
            intf['type'] = 'Ethernet'
        facts[key] = intf
    return facts