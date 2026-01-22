import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_add_nic(self, node, network):
    """
        Creates a network adapter that will connect to the specified network
        for the given node. Returns a boolean indicating success or not.
        """
    if isinstance(node, str):
        node_id = node
    else:
        node_id = node.id
    spec = {}
    spec['mac_type'] = 'GENERATED'
    spec['backing'] = {}
    spec['backing']['type'] = 'STANDARD_PORTGROUP'
    spec['backing']['network'] = network
    spec['start_connected'] = True
    data = json.dumps({'spec': spec})
    req = '/rest/vcenter/vm/{}/hardware/ethernet'.format(node_id)
    method = 'POST'
    resp = self._request(req, method=method, data=data)
    return resp.status