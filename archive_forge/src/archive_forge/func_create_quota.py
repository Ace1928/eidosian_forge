from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_quota(module, blade):
    """Create Filesystem User Quota"""
    changed = True
    if not module.check_mode:
        try:
            if module.params['gid']:
                blade.quotas_groups.create_group_quotas(file_system_names=[module.params['name']], gids=[module.params['gid']], quota=QuotasGroup(quota=int(human_to_bytes(module.params['quota']))))
            else:
                blade.quotas_groups.create_group_quotas(file_system_names=[module.params['name']], group_names=[module.params['gname']], quota=QuotasGroup(quota=int(human_to_bytes(module.params['quota']))))
        except Exception:
            if module.params['gid']:
                module.fail_json(msg='Failed to create quote for UID {0} on filesystem {1}.'.format(module.params['gid'], module.params['name']))
            else:
                module.fail_json(msg='Failed to create quote for groupname {0} on filesystem {1}.'.format(module.params['gname'], module.params['name']))
    module.exit_json(changed=changed)