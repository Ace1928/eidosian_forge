from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def create_tenant(self, **attrs):
    """Create a new tenant from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.identity.v2.tenant.Tenant`,
            comprised of the properties on the Tenant class.

        :returns: The results of tenant creation
        :rtype: :class:`~openstack.identity.v2.tenant.Tenant`
        """
    return self._create(_tenant.Tenant, **attrs)