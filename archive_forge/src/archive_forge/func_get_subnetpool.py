from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_subnetpool(self, name_or_id):
    """Get a subnetpool by name or ID.

        :param name_or_id: Name or ID of the subnetpool.

        :returns: A network ``Subnetpool`` object if found, else None.
        """
    return self.network.find_subnet_pool(name_or_id=name_or_id, ignore_missing=True)