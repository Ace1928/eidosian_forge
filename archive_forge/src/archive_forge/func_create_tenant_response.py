from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def create_tenant_response(self, tenant, parent_tenant):
    """ Creates the ansible result object from a manageiq tenant entity

        Returns:
            a dict with the tenant id, name, description, parent id,
            quota's
        """
    tenant_quotas = self.create_tenant_quotas_response(tenant['tenant_quotas'])
    try:
        ancestry = tenant['ancestry']
        tenant_parent_id = ancestry.split('/')[-1]
    except AttributeError:
        tenant_parent_id = None
    return dict(id=tenant['id'], name=tenant['name'], description=tenant['description'], parent_id=tenant_parent_id, quotas=tenant_quotas)