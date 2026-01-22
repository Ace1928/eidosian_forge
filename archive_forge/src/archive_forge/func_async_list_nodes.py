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
def async_list_nodes(self):
    """
        In this case filtering is not possible.
        Use this method when the cloud has
        a lot of vms and you want to return them all.
        """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(self._get_all_vms())
    vm_ids = [(item['vm'], item['host']) for item in result]
    interfaces = self._list_interfaces()
    return loop.run_until_complete(self._list_nodes_async(vm_ids, interfaces))