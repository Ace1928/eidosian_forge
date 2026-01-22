from openstack.common import tag
from openstack import exceptions
from openstack.network.v2 import _base
from openstack import resource
from openstack import utils
def update_external_gateways(self, session, body):
    """Update external gateways of a router.

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param dict body: The body requested to be updated on the router

        :returns: The body of the response as a dictionary.
        """
    url = utils.urljoin(self.base_path, self.id, 'update_external_gateways')
    resp = session.put(url, json=body)
    self._translate_response(resp)
    return self