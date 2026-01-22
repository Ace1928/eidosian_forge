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
def list_flavor_access(self, flavor_id):
    """List access from a private flavor for a project/tenant.

        :param string flavor_id: ID of the private flavor.

        :returns: List of dicts with flavor_id and tenant_id attributes.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    return self.compute.get_flavor_access(flavor_id)