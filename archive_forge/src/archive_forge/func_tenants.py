from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def tenants(self, **query):
    """Retrieve a generator of tenants

        :param kwargs query: Optional query parameters to be sent to limit
            the resources being returned.

        :returns: A generator of tenant instances.
        :rtype: :class:`~openstack.identity.v2.tenant.Tenant`
        """
    return self._list(_tenant.Tenant, **query)