import warnings
from novaclient import api_versions
from novaclient import base
@api_versions.wraps('2.79')
def create_server_volume(self, server_id, volume_id, device=None, tag=None, delete_on_termination=False):
    """
        Attach a volume identified by the volume ID to the given server ID

        :param server_id: The ID of the server.
        :param volume_id: The ID of the volume to attach.
        :param device: The device name (optional).
        :param tag: The tag (optional).
        :param delete_on_termination: Marked whether to delete the attached
                                      volume when the server is deleted
                                      (optional).
        :rtype: :class:`Volume`
        """
    return self._create('/servers/%s/os-volume_attachments' % server_id, VolumeManager._get_request_body_for_create(volume_id, device, tag, delete_on_termination), 'volumeAttachment')