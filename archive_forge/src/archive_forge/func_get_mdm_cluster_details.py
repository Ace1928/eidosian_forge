from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def get_mdm_cluster_details(self):
    """Get MDM cluster details
        :return: Details of MDM Cluster if existed.
        """
    try:
        mdm_cluster_details = self.powerflex_conn.system.get_mdm_cluster_details()
        if len(mdm_cluster_details) == 0:
            msg = 'MDM cluster not found'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        resp = self.get_system_details()
        if resp is not None:
            mdm_cluster_details['perfProfile'] = resp['perfProfile']
        gateway_configuration_details = self.powerflex_conn.system.get_gateway_configuration_details()
        if gateway_configuration_details is not None:
            mdm_cluster_details['mdmAddresses'] = gateway_configuration_details['mdmAddresses']
        return mdm_cluster_details
    except Exception as e:
        error_msg = 'Failed to get the MDM cluster with error {0}.'
        error_msg = error_msg.format(str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)