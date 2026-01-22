from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def edit_tenant_quota(self, tenant, current_quota, quota_key, quota_value):
    """ Update the tenant quotas in manageiq.

        Returns:
            result
        """
    if current_quota['value'] == quota_value:
        return dict(changed=False, msg='tenant quota %s already has value %s' % (quota_key, quota_value))
    else:
        url = '%s/quotas/%s' % (tenant['href'], current_quota['id'])
        resource = {'value': quota_value}
        try:
            self.client.post(url, action='edit', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to update tenant quota %s: %s' % (quota_key, str(e)))
        return dict(changed=True, msg='successfully updated tenant quota %s' % quota_key)