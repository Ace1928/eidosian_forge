import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_list_ips(self):
    """
        Lists all IPs available.

        :return: List of IP objects.
        :rtype: ``list`` of :class:`.GridscaleIp`
        """
    ips = []
    result = self._sync_request(endpoint='objects/ips/')
    for key, value in self._get_response_dict(result).items():
        ip = self._to_ip(value)
        ips.append(ip)
    return ips