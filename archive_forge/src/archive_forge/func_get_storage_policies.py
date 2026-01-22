from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_storage_policies(self, profile_manager):
    profile_ids = profile_manager.PbmQueryProfile(resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), profileCategory='REQUIREMENT')
    profiles = []
    if profile_ids:
        profiles = profile_manager.PbmRetrieveContent(profileIds=profile_ids)
    return profiles