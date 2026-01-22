from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def remove_storage_policy(self, policy, pbm_client, results):
    pbm_client.PbmDelete(profileId=[policy.profileId])
    self.format_results_and_exit(results, policy, True)