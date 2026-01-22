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
def manage_share(self, protocol, export_path, service_host, **params):
    """Manage a share.

        :param str protocol: The shared file systems protocol of this share.
        :param str export_path: The export path formatted according to the
            protocol.
        :param str service_host: The manage-share service host.
        :param kwargs params: Optional parameters to be sent. Available
            parameters include:
            * name: The user defined name of the resource.
            * share_type: The name or ID of the share type to be used to create
            the resource.
            * driver_options: A set of one or more key and value pairs, as a
            dictionary of strings, that describe driver options.
            * is_public: The level of visibility for the share.
            * description: The user defiend description of the resource.
            * share_server_id: The UUID of the share server.

        :returns: The share that was managed.
        """
    share = _share.Share()
    return share.manage(self, protocol, export_path, service_host, **params)