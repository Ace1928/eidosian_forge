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
def ex_get_size_extra_specs(self, size_id):
    """
        Get the extra_specs field of a NodeSize

        :param      size_id: ID of the size which should be used
        :type       size_id: ``str``

        :rtype: `dict`
        """
    return self.connection.request('/flavors/{}/os-extra_specs'.format(size_id)).object['extra_specs']