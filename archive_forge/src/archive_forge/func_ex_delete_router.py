import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_router(self, router):
    """
        Delete a Router

        :param router: Router which should be deleted
        :type router: :class:`OpenStack_2_Router`

        :rtype: ``bool``
        """
    resp = self.network_connection.request('{}/{}'.format('/v2.0/routers', router.id), method='DELETE')
    return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)