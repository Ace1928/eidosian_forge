import time
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
from libcloud.compute.types import Provider, NodeState

        Get the ending time of the node reservation.

        :param node: the reservation node to update
        :type  node: :class:`Node`

        :return: unix timestamp
        :rtype: ``int``
        