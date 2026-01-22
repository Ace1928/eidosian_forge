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
def resource_locks(self, **query):
    """Lists all resource locks.

        :param kwargs query: Optional query parameters to be sent to limit
            the resource locks being returned.  Available parameters include:

            * project_id: The project ID of the user that the lock is
                created for.
            * user_id: The ID of a user to filter resource locks by.
            * all_projects: list locks from all projects (Admin Only)
            * resource_id: The ID of the resource that the locks pertain to
                filter resource locks by.
            * resource_action: The action prevented by the filtered resource
                locks.
            * resource_type: The type of the resource that the locks pertain
                to filter resource locks by.
            * lock_context: The lock creator’s context to filter locks by.
            * lock_reason: The lock reason that can be used to filter resource
                locks. (Inexact search is also available with lock_reason~)
            * created_since: Search for the list of resources that were created
                after the specified date. The date is in ‘yyyy-mm-dd’ format.
            * created_before: Search for the list of resources that were
                created prior to the specified date. The date is in
                ‘yyyy-mm-dd’ format.
            * limit: The maximum number of resource locks to return.
            * offset: The offset to define start point of resource lock
                listing.
            * sort_key: The key to sort a list of shares.
            * sort_dir: The direction to sort a list of shares
            * with_count: Whether to show count in API response or not,
                default is False. This query parameter is useful with
                pagination.

        :returns: A generator of manila resource locks
        :rtype: :class:`~openstack.shared_file_system.v2.
            resource_locks.ResourceLock`
        """
    return self._list(_resource_locks.ResourceLock, **query)