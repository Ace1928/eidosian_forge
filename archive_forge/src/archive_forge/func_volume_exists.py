import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def volume_exists(self, name_or_id):
    """Check if a volume exists.

        :param name_or_id: Name or unique ID of the volume.
        :returns: True if the volume exists, else False.
        """
    return self.get_volume(name_or_id) is not None