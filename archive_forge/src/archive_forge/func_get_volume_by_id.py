import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_by_id(self, id):
    """Get a volume by ID

        :param id: ID of the volume.
        :returns: A volume ``Volume`` object if found, else None.
        """
    return self.block_storage.get_volume(id)