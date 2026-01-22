import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_list_floating_ips(self):
    """
        List floating IPs

        :rtype: ``list`` of :class:`DigitalOcean_v2_FloatingIpAddress`
        """
    return self._to_floating_ips(self._paginated_request('/v2/floating_ips', 'floating_ips'))