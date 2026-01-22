from openstack.cloud import _utils
from openstack.dns.v2._proxy import Proxy
from openstack import exceptions
from openstack import resource
def list_zones(self, filters=None):
    """List all available zones.

        :returns: A list of zones dicts.

        """
    if not filters:
        filters = {}
    return list(self.dns.zones(allow_unknown_params=True, **filters))