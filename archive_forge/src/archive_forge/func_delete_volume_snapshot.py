import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def delete_volume_snapshot(self, name_or_id=None, wait=False, timeout=None):
    """Delete a volume snapshot.

        :param name_or_id: Name or unique ID of the volume snapshot.
        :param wait: If true, waits for volume snapshot to be deleted.
        :param timeout: Seconds to wait for volume snapshot deletion. None is
            forever.

        :returns: True if deletion was successful, else False.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if wait time
            exceeded.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    volumesnapshot = self.get_volume_snapshot(name_or_id)
    if not volumesnapshot:
        return False
    self.block_storage.delete_snapshot(volumesnapshot, ignore_missing=False)
    if wait:
        self.block_storage.wait_for_delete(volumesnapshot, wait=timeout)
    return True