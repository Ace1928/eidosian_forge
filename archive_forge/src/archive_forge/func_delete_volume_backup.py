import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def delete_volume_backup(self, name_or_id=None, force=False, wait=False, timeout=None):
    """Delete a volume backup.

        :param name_or_id: Name or unique ID of the volume backup.
        :param force: Allow delete in state other than error or available.
        :param wait: If true, waits for volume backup to be deleted.
        :param timeout: Seconds to wait for volume backup deletion. None is
            forever.

        :returns: True if deletion was successful, else False.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if wait time
            exceeded.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    volume_backup = self.get_volume_backup(name_or_id)
    if not volume_backup:
        return False
    self.block_storage.delete_backup(volume_backup, ignore_missing=False, force=force)
    if wait:
        self.block_storage.wait_for_delete(volume_backup, wait=timeout)
    return True