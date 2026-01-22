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
def revert_share_to_snapshot(self, share_id, snapshot_id):
    """Reverts a share to the specified snapshot, which must be
            the most recent one known to manila.

        :param share_id: The ID of the share to revert
        :param snapshot_id: The ID of the snapshot to revert to
        :returns: Result of the ``revert``
        :rtype: ``None``
        """
    res = self._get(_share.Share, share_id)
    res.revert_to_snapshot(self, snapshot_id)