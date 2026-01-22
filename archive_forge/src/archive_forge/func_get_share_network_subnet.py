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
def get_share_network_subnet(self, share_network_id, share_network_subnet_id):
    """Lists details of a single share network subnet.

        :param share_network_id: The id of the share network associated
            with the Share Network Subnet.
        :param share_network_subnet_id: The id of the Share Network Subnet
            to retrieve.
        :returns: Details of the identified share network subnet
        :rtype:
            :class:`~openstack.shared_file_system.v2.share_network_subnet.ShareNetworkSubnet`
        """
    return self._get(_share_network_subnet.ShareNetworkSubnet, share_network_subnet_id, share_network_id=share_network_id)