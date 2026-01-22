from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def rename_mdm(self, mdm_name=None, mdm_id=None, mdm_new_name=None, cluster_details=None):
    """Rename the MDM
        :param mdm_name: Name of the MDM.
        :param mdm_id: ID of the MDM.
        :param mdm_new_name: New name of the MDM.
        :param cluster_details: Details of the MDM cluster.
        :return: True if successfully renamed.
        """
    name_or_id = mdm_id if mdm_id else mdm_name
    if mdm_name is None and mdm_id is None:
        err_msg = 'Please provide mdm_name/mdm_id to rename the MDM.'
        self.module.fail_json(msg=err_msg)
    mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
    if mdm_details is None:
        err_msg = self.not_exist_msg.format(name_or_id)
        self.module.fail_json(msg=err_msg)
    mdm_id = mdm_details['id']
    try:
        if 'name' in mdm_details and mdm_new_name != mdm_details['name'] or 'name' not in mdm_details:
            log_msg = 'Modifying the MDM name from %s to %s.' % (mdm_name, mdm_new_name)
            LOG.info(log_msg)
            if not self.module.check_mode:
                self.powerflex_conn.system.rename_mdm(mdm_id=mdm_id, mdm_new_name=mdm_new_name)
            return True
    except Exception as e:
        error_msg = 'Failed to rename the MDM {0} with error {1}.'.format(name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)