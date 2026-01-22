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
def ex_get_network_quotas(self, project_id):
    """
        Get the network quotas for a project

        :param      project_id: The ID of the project.
        :type       project_id: ``str``

        :rtype: :class:`OpenStack_2_NetworkQuota`
        """
    url = '/v2.0/quotas/%s/details.json' % project_id
    return self._to_network_quota(self.network_connection.request(url).object['quota'])