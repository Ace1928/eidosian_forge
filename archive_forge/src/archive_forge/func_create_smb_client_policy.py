from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_smb_client_policy(module, blade):
    """Create SMB Client Policy"""
    changed = True
    versions = blade.api_version.list_versions().versions
    if not module.check_mode:
        res = blade.post_smb_client_policies(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to create SMB client policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if not module.params['enabled']:
            res = blade.patch_smb_client_policies(policy=SmbClientPolicy(enabled=False), names=[module.params['name']])
            if res.status_code != 200:
                blade.delete_smb_client_policies(names=[module.params['name']])
                module.fail_json(msg='Failed to create SMB client policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if not module.params['client']:
            module.fail_json(msg='client is required to create a new rule')
        else:
            if SMB_ENCRYPT_API_VERSION in versions:
                rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], encryption=module.params['smb_encryption'])
            else:
                rule = SmbClientPolicyRule(client=module.params['client'], access=module.params['access'], permission=module.params['permission'])
            res = blade.post_smb_client_policies_rules(policy_names=[module.params['name']], rule=rule)
            if res.status_code != 200:
                module.fail_json(msg='Failed to rule for policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)