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
def ex_add_server_group(self, name, policy, rules=[]):
    """
        Add a Server Group

        :param name: Server Group Name.
        :type name: ``str``
        :param policy: Server Group policy.
        :type policy: ``str``
        :param rules: Server Group rules.
        :type rules: ``list``

        :rtype: ``bool``
        """
    data = {'name': name}
    if rules:
        data['rules'] = rules
    try:
        data['policy'] = policy
        response = self.connection.request('/os-server-groups', method='POST', data={'server_group': data}).object
    except BaseHTTPError:
        del data['policy']
        data['policies'] = [policy]
        response = self.connection.request('/os-server-groups', method='POST', data={'server_group': data}).object
    return self._to_server_group(response['server_group'])