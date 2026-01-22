import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_id(self, name_or_id):
    """Get ID of a volume.

        :param name_or_id: Name or unique ID of the volume.
        :returns: The ID of the volume if found, else None.
        """
    volume = self.get_volume(name_or_id)
    if volume:
        return volume['id']
    return None