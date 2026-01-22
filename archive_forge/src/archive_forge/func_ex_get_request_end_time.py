import time
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
from libcloud.compute.types import Provider, NodeState
def ex_get_request_end_time(self, node):
    """
        Get the ending time of the node reservation.

        :param node: the reservation node to update
        :type  node: :class:`Node`

        :return: unix timestamp
        :rtype: ``int``
        """
    res = self._vcl_request('XMLRPCgetRequestIds')
    time = 0
    for i in res['requests']:
        if i['requestid'] == node.id:
            time = i['end']
    return time