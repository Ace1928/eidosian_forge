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
def search_keypairs(self, name_or_id=None, filters=None):
    """Search keypairs.

        :param name_or_id:
        :param filters:
        :returns: A list of compute ``Keypair`` objects matching the search
            criteria.
        """
    keypairs = self.list_keypairs(filters=filters if isinstance(filters, dict) else None)
    return _utils._filter_list(keypairs, name_or_id, filters)