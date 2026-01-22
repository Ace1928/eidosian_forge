from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def update_storage_policy(self, policy, pbm_client, results):
    expected_description = self.params.get('description')
    expected_tags = [self.params.get('tag_name')]
    expected_tag_category = self.params.get('tag_category')
    expected_tag_id = self.format_tag_mob_id(expected_tag_category)
    expected_operator = 'NOT'
    if self.params.get('tag_affinity'):
        expected_operator = None
    needs_change = False
    if policy.description != expected_description:
        needs_change = True
    if hasattr(policy.constraints, 'subProfiles'):
        for subprofile in policy.constraints.subProfiles:
            tag_constraints = self.get_tag_constraints(subprofile.capability)
            if tag_constraints['id'] == expected_tag_id:
                if tag_constraints['values'] != expected_tags:
                    needs_change = True
            else:
                needs_change = True
            if tag_constraints['operator'] != expected_operator:
                needs_change = True
    else:
        needs_change = True
    if needs_change:
        pbm_client.PbmUpdate(profileId=policy.profileId, updateSpec=self.create_mob_pbm_update_spec(expected_tag_id, expected_operator, expected_tags, expected_tag_category, expected_description))
    self.format_results_and_exit(results, policy, needs_change)