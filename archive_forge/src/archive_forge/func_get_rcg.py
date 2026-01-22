from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_rcg(self, rcg_name=None, rcg_id=None):
    """Get rcg details
            :param rcg_name: Name of the RCG
            :param rcg_id: ID of the RCG
            :return: RCG details
        """
    name_or_id = rcg_id if rcg_id else rcg_name
    try:
        rcg_details = None
        if rcg_id:
            rcg_details = self.powerflex_conn.replication_consistency_group.get(filter_fields={'id': rcg_id})
        if rcg_name:
            rcg_details = self.powerflex_conn.replication_consistency_group.get(filter_fields={'name': rcg_name})
        if rcg_details:
            rcg_details[0]['statistics'] = self.powerflex_conn.replication_consistency_group.get_statistics(rcg_details[0]['id'])
            rcg_details[0].pop('links', None)
            self.append_protection_domain_name(rcg_details[0])
            return rcg_details[0]
    except Exception as e:
        errormsg = 'Failed to get the replication consistency group {0} with error {1}'.format(name_or_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)