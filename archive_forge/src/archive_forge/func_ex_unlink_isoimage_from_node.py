import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_unlink_isoimage_from_node(self, node, isoimage):
    """
        unlink isoimages from server

        :param node: node you want to unlink the image from
        :type node: ``object``

        :param isoimage: isoimage you want to unlink
        :type isoimage: ``object``

        :return: None -> success
        :rtype: ``None``
        """
    result = self._sync_request(endpoint='objects/servers/{}/isoimages/{}'.format(node.id, isoimage.id), method='DELETE')
    return result