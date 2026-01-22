from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def set_consistency(self, rcg_id, rcg_details, is_consistent):
    """Set rcg to specified mode
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details.
            :param is_consistent: RCG consistency.
            :return: Boolean indicates if set consistency is successful
        """
    try:
        if is_consistent and rcg_details['currConsistMode'].lower() not in ('consistent', 'consistentpending'):
            if not self.module.check_mode:
                self.powerflex_conn.replication_consistency_group.set_as_consistent(rcg_id)
            return True
        elif not is_consistent and rcg_details['currConsistMode'].lower() not in ('inconsistent', 'inconsistentpending'):
            if not self.module.check_mode:
                self.powerflex_conn.replication_consistency_group.set_as_inconsistent(rcg_id)
            return True
    except Exception as e:
        errormsg = 'Modifying consistency of replication consistency group failed with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)