import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def list_volume_types(self, get_extra=None):
    """List all available volume types.

        :param get_extra: **DEPRECATED** This parameter no longer does
            anything.
        :returns: A list of volume ``Type`` objects.
        """
    if get_extra is not None:
        warnings.warn("the 'get_extra' argument is deprecated and no longer does anything; consider removing it from calls", os_warnings.OpenStackDeprecationWarning)
    return list(self.block_storage.types())