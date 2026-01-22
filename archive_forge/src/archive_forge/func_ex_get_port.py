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
def ex_get_port(self, port_interface_id):
    """
        Retrieve the OpenStack_2_PortInterface with the given ID

        :param      port_interface_id: ID of the requested port
        :type       port_interface_id: str

        :return: :class:`OpenStack_2_PortInterface`
        """
    response = self.network_connection.request('/v2.0/ports/{}'.format(port_interface_id), method='GET')
    return self._to_port(response.object['port'])