from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_target_volume_access_mode(self, rcg_id, target_volume_access_mode):
    """Modify target volume access mode
            :param rcg_id: Unique identifier of the RCG.
            :param target_volume_access_mode: Target volume access mode.
            :return: Boolean indicates if modify operation is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.replication_consistency_group.modify_target_volume_access_mode(rcg_id, target_volume_access_mode)
        return True
    except Exception as e:
        errormsg = 'Modify target volume access mode for replication consistency group {0} failed with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)