from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_firewall_by_name(self):
    rule = {}
    item = 0
    for firewall in self.firewalls:
        for firewall_name in self.module.params['firewall']:
            if firewall_name in firewall['name']:
                rule[item] = {}
                rule[item].update(firewall)
                item += 1
    if len(rule) > 0:
        return rule
    return None