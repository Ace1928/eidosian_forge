from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_s3acc(module):
    """Update Object Store Account"""
    changed = False
    public = False
    blade = get_system(module)
    acc_settings = list(blade.get_object_store_accounts(names=[module.params['name']]).items)[0]
    if getattr(acc_settings, 'public_access_config', None):
        public = True
        current_account = {'hard_limit': acc_settings.hard_limit_enabled, 'default_hard_limit': acc_settings.bucket_defaults.hard_limit_enabled, 'quota': str(acc_settings.quota_limit), 'default_quota': str(acc_settings.bucket_defaults.quota_limit), 'block_new_public_policies': acc_settings.public_access_config.block_new_public_policies, 'block_public_access': acc_settings.public_access_config.block_public_access}
    else:
        current_account = {'hard_limit': acc_settings.hard_limit_enabled, 'default_hard_limit': acc_settings.bucket_defaults.hard_limit_enabled, 'quota': str(acc_settings.quota_limit), 'default_quota': str(acc_settings.bucket_defaults.quota_limit)}
    if current_account['quota'] == 'None':
        current_account['quota'] = ''
    if current_account['default_quota'] == 'None':
        current_account['default_quota'] = ''
    if module.params['quota'] is None:
        module.params['quota'] = current_account['quota']
    if module.params['default_quota'] is None:
        module.params['default_quota'] = current_account['default_quota']
    if not module.params['default_quota']:
        module.params['default_quota'] = ''
    if not module.params['quota']:
        quota = ''
    else:
        quota = str(human_to_bytes(module.params['quota']))
    if not module.params['default_quota']:
        default_quota = ''
    else:
        default_quota = str(human_to_bytes(module.params['default_quota']))
    if module.params['hard_limit'] is None:
        hard_limit = current_account['hard_limit']
    else:
        hard_limit = module.params['hard_limit']
    if module.params['default_hard_limit'] is None:
        default_hard_limit = current_account['default_hard_limit']
    else:
        default_hard_limit = module.params['default_hard_limit']
    if public:
        if module.params['block_new_public_policies'] is None:
            new_public_policies = current_account['block_new_public_policies']
        else:
            new_public_policies = module.params['block_new_public_policies']
        if module.params['block_public_access'] is None:
            public_access = current_account['block_public_access']
        else:
            public_access = module.params['block_public_access']
        new_account = {'hard_limit': hard_limit, 'default_hard_limit': default_hard_limit, 'quota': quota, 'default_quota': default_quota, 'block_new_public_policies': new_public_policies, 'block_public_access': public_access}
    else:
        new_account = {'hard_limit': module.params['hard_limit'], 'default_hard_limit': module.params['default_hard_limit'], 'quota': quota, 'default_quota': default_quota}
    if new_account != current_account:
        changed = True
        if not module.check_mode:
            osa = ObjectStoreAccountPatch(hard_limit_enabled=new_account['hard_limit'], quota_limit=new_account['quota'], bucket_defaults=BucketDefaults(hard_limit_enabled=new_account['default_hard_limit'], quota_limit=new_account['default_quota']))
            res = blade.patch_object_store_accounts(object_store_account=osa, names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to update account {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)