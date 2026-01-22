from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network
def parse_inet_line(self, words, current_if, ips):
    if words[1] == 'alias':
        del words[1]
    address = {'address': words[1]}
    if '/' in address['address']:
        ip_address, cidr_mask = address['address'].split('/')
        address['address'] = ip_address
        netmask_length = int(cidr_mask)
        netmask_bin = (1 << 32) - (1 << 32 >> int(netmask_length))
        address['netmask'] = socket.inet_ntoa(struct.pack('!L', netmask_bin))
        if len(words) > 5:
            address['broadcast'] = words[3]
    else:
        try:
            netmask_idx = words.index('netmask') + 1
        except ValueError:
            netmask_idx = 3
        if re.match('([0-9a-f]){8}$', words[netmask_idx]):
            netmask = '0x' + words[netmask_idx]
        else:
            netmask = words[netmask_idx]
        if netmask.startswith('0x'):
            address['netmask'] = socket.inet_ntoa(struct.pack('!L', int(netmask, base=16)))
        else:
            address['netmask'] = netmask
    address_bin = struct.unpack('!L', socket.inet_aton(address['address']))[0]
    netmask_bin = struct.unpack('!L', socket.inet_aton(address['netmask']))[0]
    address['network'] = socket.inet_ntoa(struct.pack('!L', address_bin & netmask_bin))
    if 'broadcast' not in address:
        try:
            broadcast_idx = words.index('broadcast') + 1
        except ValueError:
            address['broadcast'] = socket.inet_ntoa(struct.pack('!L', address_bin | ~netmask_bin & 4294967295))
        else:
            address['broadcast'] = words[broadcast_idx]
    if not words[1].startswith('127.'):
        ips['all_ipv4_addresses'].append(address['address'])
    current_if['ipv4'].append(address)