import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.compute.types import Provider, NodeState
def ex_node_by_uuid(self, uuid):
    """
        :param str ex_user_data: A valid uuid that references your existing
            cloudscale.ch server.
        :type       ex_user_data:  ``str``

        :return: The server node you asked for.
        :rtype: :class:`Node`
        """
    res = self.connection.request(self._get_server_url(uuid))
    return self._to_node(res.object)