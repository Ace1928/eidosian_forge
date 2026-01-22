from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def cluster_reduce_list(self, mdm, cluster_details):
    """Whether MDM cluster reduction is required or not.
        """
    remove_secondary = []
    remove_tb = []
    remove_secondary = self.gather_secondarys_ids(mdm, cluster_details)
    for node in mdm:
        name_or_id = node['mdm_name'] if node['mdm_name'] else node['mdm_id']
        if node['mdm_type'] == 'TieBreaker' and node['mdm_id'] is not None:
            mdm_details = self.is_mdm_name_id_exists(mdm_id=node['mdm_id'], cluster_details=cluster_details)
            if mdm_details is None:
                err_msg = self.not_exist_msg.format(name_or_id)
                self.module.fail_json(msg=err_msg)
            remove_tb.append(mdm_details['id'])
        elif node['mdm_type'] == 'TieBreaker' and node['mdm_name'] is not None:
            mdm_details = self.is_mdm_name_id_exists(mdm_name=node['mdm_name'], cluster_details=cluster_details)
            if mdm_details is None:
                err_msg = self.not_exist_msg.format(name_or_id)
                self.module.fail_json(msg=err_msg)
            else:
                remove_tb.append(mdm_details['id'])
    log_msg = 'Reduce List are: %s, %s.' % (remove_secondary, remove_tb)
    LOG.debug(log_msg)
    return (remove_secondary, remove_tb)