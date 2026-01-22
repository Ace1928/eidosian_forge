from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def format_results_and_exit(self, results, policy, changed):
    results['vmware_vm_storage_policy'] = self.format_profile(policy)
    results['changed'] = changed
    self.module.exit_json(**results)