from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_targets_dict(blade):
    targets_info = {}
    targets = blade.targets.list_targets()
    for target in range(0, len(targets.items)):
        target_name = targets.items[target].name
        targets_info[target_name] = {'address': targets.items[target].address, 'status': targets.items[target].status, 'status_details': targets.items[target].status_details}
    return targets_info