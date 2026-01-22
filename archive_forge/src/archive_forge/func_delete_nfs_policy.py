from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_nfs_policy(module, blade):
    """Delete NFS Export Policy, or Rule

    If client is provided then delete the client rule if it exists.
    """
    changed = False
    policy_delete = True
    if module.params['client']:
        policy_delete = False
        res = blade.get_nfs_export_policies_rules(policy_names=[module.params['name']], filter="client='" + module.params['client'] + "'")
        if res.status_code == 200:
            if res.total_item_count == 0:
                pass
            elif res.total_item_count == 1:
                rule = list(res.items)[0]
                if module.params['client'] == rule.client:
                    changed = True
                    if not module.check_mode:
                        res = blade.delete_nfs_export_policies_rules(names=[rule.name])
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to delete rule for client {0} in policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
            else:
                rules = list(res.items)
                for cli in range(0, len(rules)):
                    if rules[cli].client == '*':
                        changed = True
                        if not module.check_mode:
                            res = blade.delete_nfs_export_policies_rules(names=[rules[cli].name])
                            if res.status_code != 200:
                                module.fail_json(msg='Failed to delete rule for client {0} in policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
    if policy_delete:
        changed = True
        if not module.check_mode:
            res = blade.delete_nfs_export_policies(names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete export policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)