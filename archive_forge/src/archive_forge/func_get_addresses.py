from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_addresses(self, data):
    """Expose IP addresses as their own property allowing users extend to additional tasks"""
    _data = data
    for k, v in data.items():
        setattr(self, k, v)
    networks = _data['droplet']['networks']
    for network in networks.get('v4', []):
        if network['type'] == 'public':
            _data['ip_address'] = network['ip_address']
        else:
            _data['private_ipv4_address'] = network['ip_address']
    for network in networks.get('v6', []):
        if network['type'] == 'public':
            _data['ipv6_address'] = network['ip_address']
        else:
            _data['private_ipv6_address'] = network['ip_address']
    return _data