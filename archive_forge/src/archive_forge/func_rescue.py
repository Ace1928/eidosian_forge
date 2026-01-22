import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def rescue(self, session, admin_pass=None, image_ref=None):
    """Rescue the server.

        This is admin-only by default.

        :param session: The session to use for making this request.
        :param admin_pass: A new admin password to set on the rescued server.
            (Optional)
        :param image_ref: The image to use when rescuing the server. If not
            provided, the server will use the existing image. (Optional)
        :returns: None
        """
    body: ty.Dict[str, ty.Any] = {'rescue': {}}
    if admin_pass is not None:
        body['rescue']['adminPass'] = admin_pass
    if image_ref is not None:
        body['rescue']['rescue_image_ref'] = image_ref
    self._action(session, body)