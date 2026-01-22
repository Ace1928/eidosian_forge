from openstack import exceptions
from openstack import resource
from openstack import utils
def replace_metadata(self, session, metadata=None):
    """Replaces all metadata key value pairs on the resource.

        :param session: The session to use for making this request.
        :param dict metadata: Dictionary with key-value pairs
        :param bool replace: Replace all resource metadata with the new object
            or merge new and existing.
        """
    return self.set_metadata(session, metadata, replace=True)