from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_object_store_accounts_dict(blade):
    account_info = {}
    accounts = list(blade.get_object_store_accounts().items)
    for account in range(0, len(accounts)):
        acc_name = accounts[account].name
        account_info[acc_name] = {'object_count': accounts[account].object_count, 'data_reduction': accounts[account].space.data_reduction, 'snapshots_space': accounts[account].space.snapshots, 'total_physical_space': accounts[account].space.total_physical, 'unique_space': accounts[account].space.unique, 'virtual_space': accounts[account].space.virtual, 'total_provisioned_space': getattr(accounts[account].space, 'total_provisioned', None), 'available_provisioned_space': getattr(accounts[account].space, 'available_provisioned', None), 'available_ratio': getattr(accounts[account].space, 'available_ratio', None), 'destroyed_space': getattr(accounts[account].space, 'destroyed', None), 'destroyed_virtual_space': getattr(accounts[account].space, 'destroyed_virtual', None), 'quota_limit': getattr(accounts[account], 'quota_limit', None), 'hard_limit_enabled': getattr(accounts[account], 'hard_limit_enabled', None), 'total_provisioned': getattr(accounts[account].space, 'total_provisioned', None), 'users': {}}
        try:
            account_info[acc_name]['bucket_defaults'] = {'hard_limit_enabled': accounts[account].bucket_defaults.hard_limit_enabled, 'quota_limit': accounts[account].bucket_defaults.quota_limit}
        except AttributeError:
            pass
        try:
            account_info[acc_name]['public_access_config'] = {'block_new_public_policies': accounts[account].public_access_config.block_new_public_policies, 'block_public_access': accounts[account].public_access_config.block_public_access}
        except AttributeError:
            pass
        acc_users = list(blade.get_object_store_users(filter='name="' + acc_name + '/*"').items)
        for acc_user in range(0, len(acc_users)):
            user_name = acc_users[acc_user].name.split('/')[1]
            account_info[acc_name]['users'][user_name] = {'keys': [], 'policies': []}
            if blade.get_object_store_access_keys(filter='user.name="' + acc_users[acc_user].name + '"').total_item_count != 0:
                access_keys = list(blade.get_object_store_access_keys(filter='user.name="' + acc_users[acc_user].name + '"').items)
                for key in range(0, len(access_keys)):
                    account_info[acc_name]['users'][user_name]['keys'].append({'name': access_keys[key].name, 'enabled': bool(access_keys[key].enabled)})
            if blade.get_object_store_access_policies_object_store_users(member_names=[acc_users[acc_user].name]).total_item_count != 0:
                policies = list(blade.get_object_store_access_policies_object_store_users(member_names=[acc_users[acc_user].name]).items)
                for policy in range(0, len(policies)):
                    account_info[acc_name]['users'][user_name]['policies'].append(policies[policy].policy.name)
    return account_info