from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_smb_client_policy(module, blade):
    """Update SMB Client Policy Rule"""
    changed = False
    versions = blade.api_version.list_versions().versions
    if module.params['client']:
        current_policy_rule = blade.get_smb_client_policies_rules(policy_names=[module.params['name']], filter="client='" + module.params['client'] + "'")
        if current_policy_rule.status_code == 200 and current_policy_rule.total_item_count == 0:
            if SMB_ENCRYPT_API_VERSION in versions:
                rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], encryption=module.params['smb_encryption'])
            else:
                rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'])
            changed = True
            if not module.check_mode:
                if module.params['before_rule']:
                    before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                    res = blade.post_smb_client_policies_rules(policy_names=[module.params['name']], rule=rule, before_rule_name=before_name)
                else:
                    res = blade.post_smb_client_policies_rules(policy_names=[module.params['name']], rule=rule)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to create rule for client {0} in policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
        else:
            rules = list(current_policy_rule.items)
            cli_count = None
            done = False
            if module.params['client'] == '*':
                for cli in range(0, len(rules)):
                    if rules[cli].client == '*':
                        cli_count = cli
                if not cli_count:
                    if SMB_ENCRYPT_API_VERSION in versions:
                        rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], encryption=module.params['smb_encryption'])
                    else:
                        rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'])
                    done = True
                    changed = True
                    if not module.check_mode:
                        if module.params['before_rule']:
                            res = blade.post_smb_client_policies_rules(policy_names=[module.params['name']], rule=rule, before_rule_name=(module.params['name'] + '.' + str(module.params['before_rule']),))
                        else:
                            res = blade.post_smb_client_policies_rules(policy_names=[module.params['name']], rule=rule)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to create rule for client {0} in policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
            if not done:
                old_policy_rule = rules[0]
                if SMB_ENCRYPT_API_VERSION in versions:
                    current_rule = {'client': sorted(old_policy_rule.client), 'permission': sorted(old_policy_rule.permission), 'encryption': old_policy_rule.encryption}
                else:
                    current_rule = {'client': sorted(old_policy_rule.client), 'permission': sorted(old_policy_rule.permission)}
                if SMB_ENCRYPT_API_VERSION in versions:
                    if module.params['smb_encryption']:
                        new_encryption = module.params['smb_encryption']
                    else:
                        new_encryption = current_rule['encryption']
                if module.params['permission']:
                    new_permission = sorted(module.params['permission'])
                else:
                    new_permission = sorted(current_rule['permission'])
                if module.params['client']:
                    new_client = sorted(module.params['client'])
                else:
                    new_client = sorted(current_rule['client'])
                if SMB_ENCRYPT_API_VERSION in versions:
                    new_rule = {'client': new_client, 'permission': new_permission, 'encryption': new_encryption}
                else:
                    new_rule = {'client': new_client, 'permission': new_permission}
                if current_rule != new_rule:
                    changed = True
                    if not module.check_mode:
                        if SMB_ENCRYPT_API_VERSION in versions:
                            rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'], encryption=module.params['smb_encryption'])
                        else:
                            rule = SmbClientPolicyRule(client=module.params['client'], permission=module.params['permission'])
                        res = blade.patch_smb_client_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=rule)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to update SMB client rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
                if module.params['before_rule'] and module.params['before_rule'] != old_policy_rule.index:
                    changed = True
                    if not module.check_mode:
                        before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                        res = blade.patch_smb_client_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=SmbClientPolicyRule(), before_rule_name=before_name)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to move SMB client rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
    current_policy = list(blade.get_smb_client_policies(names=[module.params['name']]).items)[0]
    if current_policy.enabled != module.params['enabled']:
        changed = True
        if not module.check_mode:
            res = blade.patch_smb_client_policies(policy=SmbClientPolicy(enabled=module.params['enabled']), names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to change state of SMB client policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)