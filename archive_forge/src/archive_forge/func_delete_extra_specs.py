from openstack import exceptions
from openstack import resource
from openstack import utils
def delete_extra_specs(self, session, keys):
    """Delete extra specs.

        .. note::

            This method will do a HTTP DELETE request for every key in keys.

        :param session: The session to use for this request.
        :param list keys: The keys to delete.
        :returns: ``None``
        """
    for key in keys:
        self._extra_specs(session.delete, key=key, delete=True)