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
def ex_list_datacenters(self, ex_filter_folders=None, ex_filter_names=None, ex_filter_datacenters=None):
    req = '/rest/vcenter/datacenter'
    kwargs = {'filter.folders': ex_filter_folders, 'filter.names': ex_filter_names, 'filter.datacenters': ex_filter_datacenters}
    params = {}
    for param, value in kwargs.items():
        if value:
            params[param] = value
    result = self._request(req, params=params)
    to_return = [{'name': item['name'], 'id': item['datacenter']} for item in result.object['value']]
    return to_return