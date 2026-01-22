import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def remove_fixed_ip(self, session, address):
    """Remove a fixed IP from the server.

        This is effectively an alias from removing a port from the server.

        :param session: The session to use for making this request.
        :param network_id: The address to remove from the server.
        :returns: None
        """
    body = {'removeFixedIp': {'address': address}}
    self._action(session, body)