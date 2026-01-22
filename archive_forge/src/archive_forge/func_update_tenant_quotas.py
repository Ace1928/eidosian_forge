from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def update_tenant_quotas(self, tenant, quotas):
    """ Creates the tenant quotas in manageiq.

        Returns:
            dict with `msg` and `changed`
        """
    changed = False
    messages = []
    for quota_key, quota_value in quotas.items():
        current_quota_filtered = self.tenant_quota(tenant, quota_key)
        if current_quota_filtered:
            current_quota = current_quota_filtered[0]
        else:
            current_quota = None
        if quota_value:
            if quota_key in ['storage_allocated', 'mem_allocated']:
                quota_value_int = int(quota_value) * 1024 * 1024 * 1024
            else:
                quota_value_int = int(quota_value)
            if current_quota:
                res = self.edit_tenant_quota(tenant, current_quota, quota_key, quota_value_int)
            else:
                res = self.create_tenant_quota(tenant, quota_key, quota_value_int)
        elif current_quota:
            res = self.delete_tenant_quota(tenant, current_quota)
        else:
            res = dict(changed=False, msg="tenant quota '%s' does not exist" % quota_key)
        if res['changed']:
            changed = True
        messages.append(res['msg'])
    return dict(changed=changed, msg=', '.join(messages))