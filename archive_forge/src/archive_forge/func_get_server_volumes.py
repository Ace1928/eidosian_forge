import warnings
from novaclient import api_versions
from novaclient import base
def get_server_volumes(self, server_id):
    """
        Get a list of all the attached volumes for the given server ID

        :param server_id: The ID of the server
        :rtype: list of :class:`Volume`
        """
    return self._list('/servers/%s/os-volume_attachments' % server_id, 'volumeAttachments')