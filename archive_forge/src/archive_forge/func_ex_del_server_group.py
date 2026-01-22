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
def ex_del_server_group(self, server_group):
    """
        Delete a Server Group

        :param server_group: Server Group which should be deleted
        :type server_group: :class:`OpenStack_2_ServerGroup`

        :rtype: ``bool``
        """
    resp = self.connection.request('/os-server-groups/%s' % server_group.id, method='DELETE')
    return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)