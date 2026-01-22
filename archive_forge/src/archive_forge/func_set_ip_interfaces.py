from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def set_ip_interfaces(self, line4):
    ipv4_addresses = list()
    for line in line4:
        ipv4Split = line.split()
        if 'Ethernet' in ipv4Split[0]:
            ipv4_addresses.append(ipv4Split[1])
        if 'mgmt' in ipv4Split[0]:
            ipv4_addresses.append(ipv4Split[1])
        if 'po' in ipv4Split[0]:
            ipv4_addresses.append(ipv4Split[1])
        if 'loopback' in ipv4Split[0]:
            ipv4_addresses.append(ipv4Split[1])
    return ipv4_addresses