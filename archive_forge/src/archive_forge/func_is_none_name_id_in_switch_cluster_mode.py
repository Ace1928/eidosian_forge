from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def is_none_name_id_in_switch_cluster_mode(self, mdm):
    """ Check whether mdm dict have mdm_name and mdm_id or not"""
    for node in mdm:
        if node['mdm_id'] and node['mdm_name']:
            msg = 'parameters are mutually exclusive: mdm_name|mdm_id'
            self.module.fail_json(msg=msg)