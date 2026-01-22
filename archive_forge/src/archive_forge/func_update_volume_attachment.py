import warnings
from openstack.block_storage.v3 import volume as _volume
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import image as _image
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import limits
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import quota_set as _quota_set
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_diagnostics as _server_diagnostics
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import server_remote_console as _src
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack.network.v2 import security_group as _sg
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def update_volume_attachment(self, server, volume, volume_id=None, **attrs):
    """Update a volume attachment

        Note that the underlying API expects a volume ID, not a volume
        attachment ID. There is currently no way to update volume attachments
        by their own ID.

        :param server: The value can be either the ID of a server or a
            :class:`~openstack.compute.v2.server.Server` instance that the
            volume is attached to.
        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v3.volume.Volume` instance.
        :param volume_id: The ID of a volume to swap to. If this is not
            specified, we will default to not swapping the volume.
        :param attrs: The attributes to update on the volume attachment
            represented by ``volume_attachment``.

        :returns: ``None``
        """
    new_volume_id = volume_id
    server_id = resource.Resource._get_id(server)
    volume_id = resource.Resource._get_id(volume)
    if new_volume_id is None:
        new_volume_id = volume_id
    return self._update(_volume_attachment.VolumeAttachment, id=volume_id, server_id=server_id, volume_id=new_volume_id, **attrs)