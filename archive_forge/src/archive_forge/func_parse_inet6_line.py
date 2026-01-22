from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network
def parse_inet6_line(self, words, current_if, ips):
    address = {'address': words[1]}
    if '/' in address['address']:
        ip_address, cidr_mask = address['address'].split('/')
        address['address'] = ip_address
        address['prefix'] = cidr_mask
        if len(words) > 5:
            address['scope'] = words[5]
    else:
        if len(words) >= 4 and words[2] == 'prefixlen':
            address['prefix'] = words[3]
        if len(words) >= 6 and words[4] == 'scopeid':
            address['scope'] = words[5]
    localhost6 = ['::1', '::1/128', 'fe80::1%lo0']
    if address['address'] not in localhost6:
        ips['all_ipv6_addresses'].append(address['address'])
    current_if['ipv6'].append(address)