from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_mac_list(original_allowed, new_mac_list, state):
    if state == 'deleted':
        return [entry for entry in original_allowed if entry not in new_mac_list]
    if state == 'merged':
        return original_allowed + list(set(new_mac_list) - set(original_allowed))
    return new_mac_list