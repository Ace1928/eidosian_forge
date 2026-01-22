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
def ex_set_server_name(self, node, name):
    """
        Sets the Node's name.

        :param      node: Node
        :type       node: :class:`Node`

        :param      name: The name of the server.
        :type       name: ``str``

        :rtype: :class:`Node`
        """
    return self._update_node(node, name=name)