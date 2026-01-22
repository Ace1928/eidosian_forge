from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def tenant_quotas(self, tenant):
    """ Search for tenant quotas object by tenant.
        Returns:
            the quotas for the tenant, or None if no tenant quotas were not found.
        """
    tenant_quotas = self.client.get('%s/quotas?expand=resources' % tenant['href'])
    return tenant_quotas['resources']