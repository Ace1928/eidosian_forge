from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_interfaces_description(self, data, interfaces):
    for line in data.split('\n'):
        match = re.match('(\\d\\/\\d+)\\s+(\\w+)\\s+(\\w+)', line)
        if match:
            name = match.group(1)
            interface = {}
            interface['operstatus'] = match.group(2)
            interface['lineprotocol'] = match.group(3)
            interface['description'] = line[30:]
            interfaces[name] = interface