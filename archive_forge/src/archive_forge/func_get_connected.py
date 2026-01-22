from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_connected(module, blade):
    connected_blades = blade.array_connections.list_array_connections()
    for target in range(0, len(connected_blades.items)):
        if (connected_blades.items[target].remote.name == module.params['target'] or connected_blades.items[target].management_address == module.params['target']) and connected_blades.items[target].status in ['connected', 'connecting', 'partially_connected']:
            return connected_blades.items[target].remote.name
    connected_targets = blade.targets.list_targets()
    for target in range(0, len(connected_targets.items)):
        if connected_targets.items[target].name == module.params['target'] and connected_targets.items[target].status in ['connected', 'connecting', 'partially_connected']:
            return connected_targets.items[target].name
    return None