from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def find_mdm_in_standby(self, mdm_name=None, mdm_id=None, cluster_details=None, name_or_id=None):
    """Whether MDM exists with mdm_name or id in standby MDMs"""
    if 'standbyMDMs' in cluster_details:
        for mdm in cluster_details['standbyMDMs']:
            if 'name' in mdm and mdm_name == mdm['name'] or mdm_id == mdm['id']:
                LOG.info('MDM %s found in standby MDM.', name_or_id)
                return mdm