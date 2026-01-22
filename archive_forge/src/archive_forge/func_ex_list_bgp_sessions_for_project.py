import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_list_bgp_sessions_for_project(self, ex_project_id):
    path = '/metal/v1/projects/%s/bgp/sessions' % ex_project_id
    return self.connection.request(path).object