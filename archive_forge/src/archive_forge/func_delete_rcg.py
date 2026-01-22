from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def delete_rcg(self, rcg_id):
    """Delete RCG
            :param rcg_id: Unique identifier of the RCG.
            :return: Boolean indicates if delete RCG operation is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.replication_consistency_group.delete(rcg_id=rcg_id)
        return True
    except Exception as e:
        errormsg = 'Delete replication consistency group {0} failed with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)