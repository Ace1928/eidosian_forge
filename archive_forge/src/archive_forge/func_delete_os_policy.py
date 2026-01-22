from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_os_policy(module, blade):
    """Delete Object Store Access Policy, Rule, or User

    If rule is provided then delete the rule if it exists.
    If user is provided then remove grant from user if granted.
    If no user or rule provided delete the whole policy.
    Cannot delete a policy with attached users, so delete all users
    if the force_delete option is selected.
    """
    changed = False
    policy_name = module.params['account'] + '/' + module.params['name']
    policy_delete = True
    if module.params['rule']:
        policy_delete = False
        res = blade.get_object_store_access_policies_rules(policy_names=[policy_name], names=[module.params['rule']])
        if res.status_code == 200 and res.total_item_count != 0:
            changed = True
            if not module.check_mode:
                res = blade.delete_object_store_access_policies_object_store_rules(policy_names=[policy_name], names=[module.params['rule']])
                if res.status_code != 200:
                    module.fail_json(msg='Failed to delete users from policy {0}. Error: {1} - {2}'.format(policy_name, res.errors[0].context, res.errors[0].message))
    if module.params['user']:
        member_name = module.params['account'] + '/' + module.params['user']
        policy_delete = False
        res = blade.get_object_store_access_policies_object_store_users(policy_names=[policy_name], member_names=[member_name])
        if res.status_code == 200 and res.total_item_count != 0:
            changed = True
            if not module.check_mode:
                member_name = module.params['account'] + '/' + module.params['user']
                res = blade.delete_object_store_access_policies_object_store_users(policy_names=[policy_name], member_names=[member_name])
                if res.status_code != 200:
                    module.fail_json(msg='Failed to delete users from policy {0}. Error: {1} - {2}'.format(policy_name, res.errors[0].context, res.errors[0].message))
    if policy_delete:
        if module.params['account'].lower() == 'pure:policy':
            module.fail_json(msg='System-Wide policies cannot be deleted.')
        policy_users = list(blade.get_object_store_access_policies_object_store_users(policy_names=[policy_name]).items)
        if len(policy_users) == 0:
            changed = True
            if not module.check_mode:
                res = blade.delete_object_store_access_policies(names=[policy_name])
                if res.status_code != 200:
                    module.fail_json(msg='Failed to delete policy {0}. Error: {1}'.format(policy_name, res.errors[0].message))
        elif module.params['force_delete']:
            changed = True
            if not module.check_mode:
                for user in range(0, len(policy_users)):
                    res = blade.delete_object_store_access_policies_object_store_users(member_names=[policy_users[user].member.name], policy_names=[policy_name])
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to delete user {0} from policy {1}, Error: {2}'.format(policy_users[user].member, policy_name, res.errors[0].message))
                res = blade.delete_object_store_access_policies(names=[policy_name])
                if res.status_code != 200:
                    module.fail_json(msg='Failed to delete policy {0}. Error: {1}'.format(policy_name, res.errors[0].message))
        else:
            module.fail_json(msg='Policy {0} cannot be deleted with connected users'.format(policy_name))
    module.exit_json(changed=changed)