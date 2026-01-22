from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def switchover(self, rcg_id, force):
    """Perform switchover
            :param rcg_id: Unique identifier of the RCG.
            :param force: Force switchover.
            :return: Boolean indicates if RCG switchover is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.replication_consistency_group.switchover(rcg_id, force)
        return True
    except Exception as e:
        errormsg = f'Switchover replication consistency group {rcg_id} failed with error {e}'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)