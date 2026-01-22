from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
def rename_fault_set(self, fault_set_id, new_name):
    """Perform rename operation on a fault set"""
    try:
        if not self.module.check_mode:
            self.powerflex_conn.fault_set.rename(fault_set_id=fault_set_id, name=new_name)
        return self.get_fault_set(fault_set_id=fault_set_id)
    except Exception as e:
        msg = f'Failed to rename the fault set instance with error {str(e)}'
        LOG.error(msg)
        self.module.fail_json(msg=msg)