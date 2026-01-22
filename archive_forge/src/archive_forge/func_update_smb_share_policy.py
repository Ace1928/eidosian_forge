from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_smb_share_policy(module, blade):
    """Update SMB Share Policy Rule"""
    changed = False
    if module.params['principal']:
        current_policy_rule = blade.get_smb_share_policies_rules(policy_names=[module.params['name']], filter="principal='" + module.params['principal'] + "'")
        if current_policy_rule.status_code == 200 and current_policy_rule.total_item_count == 0:
            rule = SmbSharePolicyRule(principal=module.params['principal'], change=module.params['change'], read=module.params['read'], full_control=module.params['full_control'])
            changed = True
            if not module.check_mode:
                if module.params['before_rule']:
                    before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                    res = blade.post_smb_share_policies_rules(policy_names=[module.params['name']], rule=rule, before_rule_name=before_name)
                else:
                    res = blade.post_smb_share_policies_rules(policy_names=[module.params['name']], rule=rule)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to create rule for principal {0} in policy {1}. Error: {2}'.format(module.params['principal'], module.params['name'], res.errors[0].message))
        else:
            rules = list(current_policy_rule.items)
            cli_count = None
            old_policy_rule = rules[0]
            current_rule = {'principal': sorted(old_policy_rule.principal), 'read': sorted(old_policy_rule.read), 'change': sorted(old_policy_rule.change), 'full_control': sorted(old_policy_rule.full_control)}
            if module.params['read']:
                if module.params['read'] == '':
                    new_read = ''
                else:
                    new_read = module.params['read']
            else:
                new_read = current_rule['read']
            if module.params['full_control']:
                if module.params['full_control'] == '':
                    new_full_control = ''
                else:
                    new_full_control = module.params['full_control']
            else:
                new_full_control = current_rule['full_control']
            if module.params['change']:
                if module.params['change'] == '':
                    new_change = ''
                else:
                    new_change = module.params['change']
            else:
                new_change = current_rule['change']
            if module.params['principal']:
                new_principal = module.params['principal']
            else:
                new_principal = current_rule['principal']
            new_rule = {'principal': new_principal, 'read': new_read, 'change': new_change, 'full_control': new_full_control}
            if current_rule != new_rule:
                changed = True
                if not module.check_mode:
                    rule = SmbSharePolicyRule(principal=module.params['principal'], change=module.params['change'], read=module.params['read'], full_control=module.params['full_control'])
                    res = blade.patch_smb_share_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=rule)
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to update SMB share rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
            if module.params['before_rule'] and module.params['before_rule'] != old_policy_rule.index:
                changed = True
                if not module.check_mode:
                    before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                    res = blade.patch_smb_share_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=SmbSharePolicyRule(), before_rule_name=before_name)
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to move SMB share rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
    current_policy = list(blade.get_smb_share_policies(names=[module.params['name']]).items)[0]
    if current_policy.enabled != module.params['enabled']:
        changed = True
        if not module.check_mode:
            res = blade.patch_smb_share_policies(policy=SmbSharePolicy(enabled=module.params['enabled']), names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to change state of SMB share policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)