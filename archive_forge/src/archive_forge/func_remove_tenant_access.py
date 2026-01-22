from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_tenant_access(self, session, tenant):
    """Removes flavor access to a tenant and flavor.

        :param session: The session to use for making this request.
        :param tenant:
        :returns: None
        """
    body = {'removeTenantAccess': {'tenant': tenant}}
    self._action(session, body)