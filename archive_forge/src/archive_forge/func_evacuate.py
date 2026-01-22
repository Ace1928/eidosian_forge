import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def evacuate(self, session, host=None, admin_pass=None, force=None):
    """Evacuate the server.

        :param session: The session to use for making this request.
        :param host: The host to evacuate the instance to. (Optional)
        :param admin_pass: The admin password to set on the evacuated instance.
            (Optional)
        :param force: Whether to force evacuation.
        :returns: None
        """
    body: ty.Dict[str, ty.Any] = {'evacuate': {}}
    if host is not None:
        body['evacuate']['host'] = host
    if admin_pass is not None:
        body['evacuate']['adminPass'] = admin_pass
    if force is not None:
        body['evacuate']['force'] = force
    self._action(session, body)