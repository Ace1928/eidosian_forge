from openstack.cloud import _utils
from openstack.dns.v2._proxy import Proxy
from openstack import exceptions
from openstack import resource
def search_zones(self, name_or_id=None, filters=None):
    zones = self.list_zones(filters)
    return _utils._filter_list(zones, name_or_id, filters)