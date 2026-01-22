from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def set_ipv6_interfaces(self, line4):
    ipv6_addresses = list()
    for line in line4:
        ipv6Split = line.split()
        if 'Ethernet' in ipv6Split[0]:
            ipv6_addresses.append(ipv6Split[1])
        if 'mgmt' in ipv6Split[0]:
            ipv6_addresses.append(ipv6Split[1])
        if 'po' in ipv6Split[0]:
            ipv6_addresses.append(ipv6Split[1])
        if 'loopback' in ipv6Split[0]:
            ipv6_addresses.append(ipv6Split[1])
    return ipv6_addresses