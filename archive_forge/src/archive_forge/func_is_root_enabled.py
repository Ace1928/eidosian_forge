from openstack import resource
from openstack import utils
def is_root_enabled(self, session):
    """Determine if root is enabled on an instance.

        Determine if root is enabled on this particular instance.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: ``True`` if root user is enabled for a specified database
            instance or ``False`` otherwise.
        """
    url = utils.urljoin(self.base_path, self.id, 'root')
    resp = session.get(url)
    return resp.json()['rootEnabled']