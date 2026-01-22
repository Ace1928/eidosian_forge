from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_smb_share_policy(module, blade):
    """Delete SMB Share Policy, or Rule

    If principal is provided then delete the principal rule if it exists.
    """
    changed = False
    policy_delete = True
    if module.params['principal']:
        policy_delete = False
        prin_rule = blade.get_smb_share_policies_rules(policy_names=[module.params['name']], filter="principal='" + module.params['principal'] + "'")
        if prin_rule.status_code == 200:
            rule = list(prin_rule.items)[0]
            changed = True
            if not module.check_mode:
                res = blade.delete_smb_share_policies_rules(names=[rule.name])
                if res.status_code != 200:
                    module.fail_json(msg='Failed to delete rule for principal {0} in policy {1}. Error: {2}'.format(module.params['principal'], module.params['name'], res.errors[0].message))
    if policy_delete:
        changed = True
        if not module.check_mode:
            res = blade.delete_smb_share_policies(names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete SMB share policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)