import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_list_events_for_node(self, node, include=None, page=1, per_page=10):
    path = '/metal/v1/devices/%s/events' % node.id
    params = {'include': include, 'page': page, 'per_page': per_page}
    return self.connection.request(path, params=params).object