from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_s3acc(module, blade):
    """Create Object Store Account"""
    changed = True
    versions = blade.api_version.list_versions().versions
    if not module.check_mode:
        try:
            blade.object_store_accounts.create_object_store_accounts(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Object Store Account {0}: Creation failed'.format(module.params['name']))
        if module.params['quota'] or module.params['default_quota']:
            blade2 = get_system(module)
            if not module.params['default_quota']:
                default_quota = ''
            else:
                default_quota = str(human_to_bytes(module.params['default_quota']))
            if not module.params['quota']:
                quota = ''
            else:
                quota = str(human_to_bytes(module.params['quota']))
            if not module.params['hard_limit']:
                module.params['hard_limit'] = False
            if not module.params['default_hard_limit']:
                module.params['default_hard_limit'] = False
            osa = ObjectStoreAccountPatch(hard_limit_enabled=module.params['hard_limit'], quota_limit=quota, bucket_defaults=BucketDefaults(hard_limit_enabled=module.params['default_hard_limit'], quota_limit=default_quota))
            res = blade2.patch_object_store_accounts(object_store_account=osa, names=[module.params['name']])
            if res.status_code != 200:
                blade.object_store_accounts.delete_object_store_accounts(names=[module.params['name']])
                module.fail_json(msg='Failed to set quotas correctly for account {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
        if PUBLIC_API_VERSION in versions:
            if not module.params['block_new_public_policies']:
                module.params['block_new_public_policies'] = False
            if not module.params['block_public_access']:
                module.params['block_public_access'] = False
            osa = ObjectStoreAccountPatch(public_access_config=PublicAccessConfig(block_new_public_policies=module.params['block_new_public_policies'], block_public_access=module.params['block_public_access']))
            res = blade2.patch_object_store_accounts(object_store_account=osa, names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to Public Access config correctly for account {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)