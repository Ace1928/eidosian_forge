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
def search_aggregates(self, name_or_id=None, filters=None):
    """Seach host aggregates.

        :param name: aggregate name or id.
        :param filters: a dict containing additional filters to use.

        :returns: A list of compute ``Aggregate`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    aggregates = self.list_aggregates()
    return _utils._filter_list(aggregates, name_or_id, filters)