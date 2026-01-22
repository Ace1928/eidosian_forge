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
def create_resource_lock(self, **attrs):
    """Locks a resource.

        :param dict attrs: Attributes which will be used to create
            a :class:`~openstack.shared_file_system.v2.
            resource_locks.ResourceLock`, comprised of the properties
            on the ResourceLock class. Available parameters include:

            * ``resource_id``: ID of the resource to be locked.
            * ``resource_type``: type of the resource (share, access_rule).
            * ``resource_action``: action to be locked (delete, show).
            * ``lock_reason``: reason why you're locking the resource
                (Optional).
        :returns: Details of the lock
        :rtype: :class:`~openstack.shared_file_system.v2.
            resource_locks.ResourceLock`
        """
    if attrs.get('resource_type'):
        attrs['__conflicting_attrs'] = {'resource_type': attrs.get('resource_type')}
        attrs.pop('resource_type')
    return self._create(_resource_locks.ResourceLock, **attrs)