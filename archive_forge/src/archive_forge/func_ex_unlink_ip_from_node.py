import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_unlink_ip_from_node(self, node, ip):
    """
        unlink ips from server

        :param node: node you want to unlink the ip from
        :type node: ``object``

        :param ip: the ip you want to unlink
        :type ip: ``object``

        :return: None -> success
        :rtype: ``None``
        """
    result = self._sync_request(endpoint='objects/servers/{}/ips/{}'.format(node.id, ip.id), method='DELETE')
    return result