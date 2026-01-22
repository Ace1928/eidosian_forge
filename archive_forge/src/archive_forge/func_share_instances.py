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
def share_instances(self, **query):
    """Lists all share instances.

        :param kwargs query: Optional query parameters to be sent to limit
            the share instances being returned. Available parameters include:

        * export_location_id: The export location UUID that can be used
          to filter share instances.
        * export_location_path: The export location path that can be used
          to filter share instances.

        :returns: Details of share instances resources
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_instance.ShareInstance`
        """
    return self._list(_share_instance.ShareInstance, **query)