from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_nfs_policy(module, blade):
    """Create NFS Export Policy"""
    changed = True
    if not module.check_mode:
        res = blade.post_nfs_export_policies(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to create nfs export policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if not module.params['enabled']:
            res = blade.patch_nfs_export_policies(policy=NfsExportPolicy(enabled=False), names=[module.params['name']])
            if res.status_code != 200:
                blade.delete_nfs_export_policies(names=[module.params['name']])
                module.fail_json(msg='Failed to create nfs export policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if module.params['client']:
            rule = NfsExportPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], anonuid=module.params['anonuid'], anongid=module.params['anongid'], fileid_32bit=module.params['fileid_32bit'], atime=module.params['atime'], secure=module.params['secure'], security=module.params['security'])
            res = blade.post_nfs_export_policies_rules(policy_names=[module.params['name']], rule=rule)
            if res.status_code != 200:
                module.fail_json(msg='Failed to rule for policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)