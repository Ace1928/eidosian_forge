from openstack.cloud import _utils
from openstack.dns.v2._proxy import Proxy
from openstack import exceptions
from openstack import resource
def search_recordsets(self, zone, name_or_id=None, filters=None):
    recordsets = self.list_recordsets(zone=zone)
    return _utils._filter_list(recordsets, name_or_id, filters)