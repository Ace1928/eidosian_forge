from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def is_mdm_name_id_exists(self, mdm_id=None, mdm_name=None, cluster_details=None):
    """Whether MDM exists with mdm_name or id """
    name_or_id = mdm_id if mdm_id else mdm_name
    if 'name' in cluster_details['master'] and mdm_name == cluster_details['master']['name'] or mdm_id == cluster_details['master']['id']:
        LOG.info('MDM %s is master MDM.', name_or_id)
        return cluster_details['master']
    secondary_mdm = []
    secondary_mdm = self.find_mdm_in_secondarys(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
    if secondary_mdm is not None:
        return secondary_mdm
    tb_mdm = []
    tb_mdm = self.find_mdm_in_tb(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
    if tb_mdm is not None:
        return tb_mdm
    standby_mdm = self.find_mdm_in_standby(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
    if standby_mdm is not None:
        return standby_mdm
    LOG.info('MDM %s does not exists in MDM Cluster.', name_or_id)
    return None