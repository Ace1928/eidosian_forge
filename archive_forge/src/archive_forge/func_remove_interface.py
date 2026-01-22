from openstack.common import tag
from openstack import exceptions
from openstack.network.v2 import _base
from openstack import resource
from openstack import utils
def remove_interface(self, session, **body):
    """Remove an internal interface from a logical router.

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param dict body: The body requested to be updated on the router

        :returns: The body of the response as a dictionary.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'remove_router_interface')
    resp = self._put(session, url, body)
    return resp.json()