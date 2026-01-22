import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
def ex_stop_nodes(self, nodes: List[Node]) -> bool:
    """Stops all the nodes given.

        : param nodes: A list of the nodes to stop.
        : type node: ``list`` of: class `Node`

        : rtype: ``bool``
        """
    data = {'instance_ids': [node.id for node in nodes]}
    resp = self.connection.request('/v2/instances/halt', data=json.dumps(data), method='POST')
    return resp.success()