from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_evc_configuration(self):
    """
        Check evc configuration
        Returns: 'Present' if evc enabled, else 'absent'
        """
    try:
        self.datacenter = find_datacenter_by_name(self.content, self.datacenter_name)
        if self.datacenter is None:
            self.module.fail_json(msg="Datacenter '%s' does not exist." % self.datacenter_name)
        self.cluster = self.find_cluster_by_name(cluster_name=self.cluster_name, datacenter_name=self.datacenter)
        if self.cluster is None:
            self.module.fail_json(msg="Cluster '%s' does not exist." % self.cluster_name)
        self.evcm = self.cluster.EvcManager()
        if not self.evcm:
            self.module.fail_json(msg="Unable to get EVC manager for cluster '%s'." % self.cluster_name)
        self.evc_state = self.evcm.evcState
        self.current_evc_mode = self.evc_state.currentEVCModeKey
        if not self.current_evc_mode:
            return 'absent'
        return 'present'
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to check configuration due to generic exception %s' % to_native(generic_exc))