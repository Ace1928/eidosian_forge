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
def ex_list_ports(self):
    """
        List all OpenStack_2_PortInterfaces

        https://developer.openstack.org/api-ref/network/v2/#list-ports

        :rtype: ``list`` of :class:`OpenStack_2_PortInterface`
        """
    response = self._paginated_request('/v2.0/ports', 'ports', self.network_connection)
    return [self._to_port(port) for port in response['ports']]