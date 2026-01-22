from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
def delete_fault_set(self, fault_set_id):
    """Delete the Fault Set"""
    try:
        if not self.module.check_mode:
            LOG.info(msg=f'Removing Fault Set {fault_set_id}')
            self.powerflex_conn.fault_set.delete(fault_set_id)
            LOG.info('returning None')
            return None
        return self.get_fault_set(fault_set_id=fault_set_id)
    except Exception as e:
        errormsg = f'Removing Fault Set {fault_set_id} failed with error {str(e)}'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)