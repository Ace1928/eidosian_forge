from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def set_performance_profile(self, performance_profile=None, cluster_details=None):
    """ Set the performance profile of Cluster MDMs
        :param performance_profile: Specifies the performance profile of MDMs
        :param cluster_details: Details of MDM cluster
        :return: True if updated successfully
        """
    if self.module.params['state'] == 'present' and performance_profile:
        if cluster_details['perfProfile'] != performance_profile:
            try:
                if not self.module.check_mode:
                    self.powerflex_conn.system.set_cluster_mdm_performance_profile(performance_profile=performance_profile)
                return True
            except Exception as e:
                error_msg = 'Failed to update performance profile to {0} with error {1}.'.format(performance_profile, str(e))
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        return False
    return False