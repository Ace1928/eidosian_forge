from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.network.base import Network, NetworkCollector
def assign_network_facts(self, network_facts, fsysopts_path, socket_path):
    rc, out, err = self.module.run_command([fsysopts_path, '-L', socket_path])
    network_facts['interfaces'] = []
    for i in out.split():
        if '=' in i and i.startswith('--'):
            k, v = i.split('=', 1)
            k = k[2:]
            if k == 'interface':
                v = v[5:]
                network_facts['interfaces'].append(v)
                network_facts[v] = {'active': True, 'device': v, 'ipv4': {}, 'ipv6': []}
                current_if = v
            elif k == 'address':
                network_facts[current_if]['ipv4']['address'] = v
            elif k == 'netmask':
                network_facts[current_if]['ipv4']['netmask'] = v
            elif k == 'address6':
                address, prefix = v.split('/')
                network_facts[current_if]['ipv6'].append({'address': address, 'prefix': prefix})
    return network_facts