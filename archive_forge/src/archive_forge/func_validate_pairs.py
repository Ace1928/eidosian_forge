from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def validate_pairs(self, params):
    for pair in params['pairs']:
        if pair['source_volume_id'] and pair['source_volume_name']:
            self.module.fail_json(msg='Specify either source_volume_id or source_volume_name')
        if pair['target_volume_id'] and pair['target_volume_name']:
            self.module.fail_json(msg='Specify either target_volume_id or target_volume_name')
        if pair['target_volume_name'] and params['remote_peer'] is None:
            self.module.fail_json(msg='Specify remote_peer with target_volume_name')