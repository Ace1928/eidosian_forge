from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def freeze_or_unfreeze_rcg(self, rcg_id, rcg_details, freeze):
    """Perform specified RCG action
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details.
            :param freeze: Freeze or unfreeze RCG.
            :return: Boolean indicates if RCG action is successful
        """
    if freeze and rcg_details['freezeState'].lower() == 'unfrozen':
        return self.freeze(rcg_id)
    if not freeze and rcg_details['freezeState'].lower() == 'frozen':
        return self.unfreeze(rcg_id)