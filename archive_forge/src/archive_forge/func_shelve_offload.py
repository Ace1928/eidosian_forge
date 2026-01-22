import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def shelve_offload(self, session):
    """Shelve-offload the server.

        :param session: The session to use for making this request.
        :returns: None
        """
    body = {'shelveOffload': None}
    self._action(session, body)