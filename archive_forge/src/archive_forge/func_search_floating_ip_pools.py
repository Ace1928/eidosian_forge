import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def search_floating_ip_pools(self, name=None, filters=None):
    pools = self.list_floating_ip_pools()
    return _utils._filter_list(pools, name, filters)