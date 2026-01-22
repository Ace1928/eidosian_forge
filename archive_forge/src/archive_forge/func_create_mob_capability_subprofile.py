from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def create_mob_capability_subprofile(self, tag_id, tag_operator, tags, tag_category):
    return pbm.profile.SubProfileCapabilityConstraints(subProfiles=[self.create_mob_capability_constraints_subprofile(tag_id, tag_operator, tags, tag_category)])