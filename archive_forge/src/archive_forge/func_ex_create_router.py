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
def ex_create_router(self, name, description='', admin_state_up=True, external_gateway_info=None):
    """
        Create a new Router

        :param name: Name of router which should be used
        :type name: ``str``

        :param      description: Description of the port
        :type       description: ``str``

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: ``bool``

        :param      external_gateway_info: The external gateway information
        :type       external_gateway_info: ``dict``

        :rtype: :class:`OpenStack_2_Router`
        """
    data = {'router': {'name': name or '', 'description': description or '', 'admin_state_up': admin_state_up}}
    if external_gateway_info:
        data['router']['external_gateway_info'] = external_gateway_info
    response = self.network_connection.request('/v2.0/routers', method='POST', data=data).object
    return self._to_router(response['router'])