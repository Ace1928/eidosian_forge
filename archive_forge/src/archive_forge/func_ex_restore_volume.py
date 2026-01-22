import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_restore_volume(self, snapshot):
    volume_id = snapshot.extra['volume']['href'].split('/')[-1]
    ts = snapshot.extra['timestamp']
    path = '/metal/v1/storage/{}/restore?restore_point={}'.format(volume_id, ts)
    res = self.connection.request(path, method='POST')
    return res.status == httplib.NO_CONTENT