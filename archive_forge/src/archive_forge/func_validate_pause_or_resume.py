from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def validate_pause_or_resume(self, pause, replication_pair_details, pair_id):
    if not replication_pair_details:
        self.module.fail_json(msg='Specify a valid pair_name or pair_id to perform pause or resume')
    return self.perform_pause_or_resume(pause, replication_pair_details, pair_id)