from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def create_replication_pairs(self, rcg_id, rcg_pairs, input_pairs):
    """Create replication pairs"""
    try:
        for pair in input_pairs:
            if pair['source_volume_name'] is not None:
                pair['source_volume_id'] = self.get_volume(pair['source_volume_name'])[0]['id']
            if pair['target_volume_name'] is not None:
                pair['target_volume_id'] = self.get_volume(pair['target_volume_name'], is_remote=True)[0]['id']
        pairs = find_non_existing_pairs(rcg_pairs, input_pairs)
        if not pairs:
            return False
        if not self.module.check_mode:
            for pair in pairs:
                self.powerflex_conn.replication_pair.add(source_vol_id=pair['source_volume_id'], dest_vol_id=pair['target_volume_id'], rcg_id=rcg_id, copy_type=pair['copy_type'], name=pair['name'])
        return True
    except Exception as e:
        errormsg = 'Create replication pairs failed with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)