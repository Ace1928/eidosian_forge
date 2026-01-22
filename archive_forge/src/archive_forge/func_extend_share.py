from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack import utils
def extend_share(self, session, new_size, force=False):
    """Extend the share size.

        :param float new_size: The new size of the share
            in GiB.
        :param bool force: Whether or not to use force, bypassing
            the scheduler. Requires admin privileges. Defaults to False.
        :returns: The result of the action.
        :rtype: ``None``
        """
    extend_body = {'new_size': new_size}
    if force is True:
        extend_body['force'] = True
    body = {'extend': extend_body}
    self._action(session, body)