import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def search_servers(self, name_or_id=None, filters=None, detailed=False, all_projects=False, bare=False):
    """Search servers.

        :param name_or_id:
        :param filters:
        :param detailed:
        :param all_projects:
        :param bare:
        :returns: A list of compute ``Server`` objects matching the search
            criteria.
        """
    servers = self.list_servers(detailed=detailed, all_projects=all_projects, bare=bare)
    return _utils._filter_list(servers, name_or_id, filters)