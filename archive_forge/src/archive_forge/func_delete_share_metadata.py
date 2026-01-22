from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import limit as _limit
from openstack.shared_file_system.v2 import resource_locks as _resource_locks
from openstack.shared_file_system.v2 import share as _share
from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_instance as _share_instance
from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_snapshot as _share_snapshot
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import storage_pool as _storage_pool
from openstack.shared_file_system.v2 import user_message as _user_message
def delete_share_metadata(self, share_id, keys, ignore_missing=True):
    """Deletes a single metadata item on a share, idetified by its key.

        :param share_id: The ID of the share
        :param keys: The list of share metadata keys to be deleted
        :param ignore_missing: Boolean indicating if missing keys should be ignored.

        :returns: None
        :rtype: None
        """
    share = self._get_resource(_share.Share, share_id)
    keys_failed_to_delete = []
    for key in keys:
        try:
            share.delete_metadata_item(self, key)
        except exceptions.NotFoundException:
            if not ignore_missing:
                self._connection.log.info('Key %s not found.', key)
                keys_failed_to_delete.append(key)
        except exceptions.ForbiddenException:
            self._connection.log.info('Key %s cannot be deleted.', key)
            keys_failed_to_delete.append(key)
        except exceptions.SDKException:
            self._connection.log.info('Failed to delete key %s.', key)
            keys_failed_to_delete.append(key)
    if keys_failed_to_delete:
        raise exceptions.SDKException('Some keys failed to be deleted %s' % keys_failed_to_delete)