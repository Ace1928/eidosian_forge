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
def ex_delete_ip_group(self, group_id):
    """
        Deletes the specified shared IP group.

        :param       group_id:  group id which should be used
        :type        group_id: ``str``

        :rtype: ``bool``
        """
    uri = '/shared_ip_groups/%s' % group_id
    resp = self.connection.request(uri, method='DELETE')
    return resp.status == httplib.NO_CONTENT