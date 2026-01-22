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
def user_messages(self, **query):
    """List shared file system user messages

        :param kwargs query: Optional query parameters to be sent to limit
            the messages being returned.  Available parameters include:

            * action_id: The ID of the action during which the message
              was created.
            * detail_id: The ID of the message detail.
            * limit: The maximum number of shares to return.
            * message_level: The message level.
            * offset: The offset to define start point of share or share
              group listing.
            * sort_key: The key to sort a list of messages.
            * sort_dir: The direction to sort a list of shares.
            * project_id: The ID of the project for which the message
              was created.
            * request_id: The ID of the request during which the message
              was created.
            * resource_id: The UUID of the resource for which the message
              was created.
            * resource_type: The type of the resource for which the message
              was created.

        :returns: A generator of user message resources
        :rtype:
            :class:`~openstack.shared_file_system.v2.user_message.UserMessage`
        """
    return self._list(_user_message.UserMessage, **query)