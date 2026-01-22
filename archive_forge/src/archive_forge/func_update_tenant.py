from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def update_tenant(self, tenant, **attrs):
    """Update a tenant

        :param tenant: Either the ID of a tenant or a
            :class:`~openstack.identity.v2.tenant.Tenant` instance.
        :param attrs: The attributes to update on the tenant represented
            by ``tenant``.

        :returns: The updated tenant
        :rtype: :class:`~openstack.identity.v2.tenant.Tenant`
        """
    return self._update(_tenant.Tenant, tenant, **attrs)