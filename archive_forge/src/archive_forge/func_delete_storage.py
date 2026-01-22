from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def delete_storage(idrac, module):
    names = [key.get('name') for key in module.params['volumes']]
    storage_status = idrac.config_mgr.RaidHelper.delete_virtual_disk(vd_names=names, apply_changes=not module.check_mode)
    return storage_status