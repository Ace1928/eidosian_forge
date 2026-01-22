import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def start_node(self, node):
    """
        Start virtual machine
        """
    node.state = NodeState.STARTING
    self._api_request('/machines/start', {'machineId': int(node.id)})
    node.state = NodeState.RUNNING
    return True