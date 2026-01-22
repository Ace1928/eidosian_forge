from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network
def merge_default_interface(self, defaults, interfaces, ip_type):
    if 'interface' not in defaults:
        return
    if not defaults['interface'] in interfaces:
        return
    ifinfo = interfaces[defaults['interface']]
    for item in ifinfo:
        if item != 'ipv4' and item != 'ipv6':
            defaults[item] = ifinfo[item]
    ipinfo = []
    if 'address' in defaults:
        ipinfo = [x for x in ifinfo[ip_type] if x['address'] == defaults['address']]
    if len(ipinfo) == 0:
        ipinfo = ifinfo[ip_type]
    if len(ipinfo) > 0:
        for item in ipinfo[0]:
            defaults[item] = ipinfo[0][item]