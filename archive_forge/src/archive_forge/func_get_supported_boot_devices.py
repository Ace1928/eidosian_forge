import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def get_supported_boot_devices(self, session):
    """Get supported boot devices for the node.

        :param session: The session to use for making this request.
        :returns: The HTTP response.
        """
    session = self._get_session(session)
    version = self._get_microversion(session, action='fetch')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'management', 'boot_device', 'supported')
    response = session.get(request.url, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to get supported boot devices for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
    return response.json()