from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_networks(self, filters=None):
    """List all available networks.

        :param filters: (optional) A dict of filter conditions to push down.
        :returns: A list of network ``Network`` objects.
        """
    if not self.has_service('network'):
        return []
    if not filters:
        filters = {}
    return list(self.network.networks(**filters))