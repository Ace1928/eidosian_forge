from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack import utils
def shrink_share(self, session, new_size):
    """Shrink the share size.

        :param float new_size: The new size of the share
            in GiB.
        :returns: ``None``
        """
    body = {'shrink': {'new_size': new_size}}
    self._action(session, body)