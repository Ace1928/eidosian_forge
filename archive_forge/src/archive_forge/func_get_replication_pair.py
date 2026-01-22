from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_replication_pair(self, pair_name=None, pair_id=None):
    """Get replication pair details
            :param pair_name: Name of the replication pair
            :param pair_id: ID of the replication pair
            :return: Replication pair details
        """
    name_or_id = pair_id if pair_id else pair_name
    try:
        pair_details = []
        if pair_id:
            pair_details = self.powerflex_conn.replication_pair.get(filter_fields={'id': pair_id})
        if pair_name:
            pair_details = self.powerflex_conn.replication_pair.get(filter_fields={'name': pair_name})
        if pair_details:
            pair_details[0].pop('links', None)
            pair_details[0]['localVolumeName'] = self.get_volume(pair_details[0]['localVolumeId'], filter_by_name=False)[0]['name']
            pair_details[0]['statistics'] = self.powerflex_conn.replication_pair.get_statistics(pair_details[0]['id'])
            return pair_details[0]
        return pair_details
    except Exception as e:
        errormsg = 'Failed to get the replication pair {0} with error {1}'.format(name_or_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)