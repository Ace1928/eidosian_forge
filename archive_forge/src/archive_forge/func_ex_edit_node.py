import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_edit_node(self, node, params):
    """
        Edit a node.

        :param node: Node to edit.
        :type node: :class:`libcloud.compute.base.Node`

        :param params: Node parameters to update.
        :type params: ``dict``

        :return Edited node.
        :rtype: :class:`libcloud.compute.base.Node`
        """
    data = {}
    data['name'] = node.name
    data['cpu'] = node.extra['cpus']
    data['mem'] = node.extra['memory']
    data['vnc_password'] = node.extra['vnc_password']
    nics = copy.deepcopy(node.extra.get('nics', []))
    data['nics'] = nics
    data.update(params)
    action = '/servers/%s/' % node.id
    response = self.connection.request(action=action, method='PUT', data=data).object
    node = self._to_node(data=response)
    return node