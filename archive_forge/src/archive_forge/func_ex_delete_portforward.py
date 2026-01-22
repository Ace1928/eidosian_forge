import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_portforward(self, portforward):
    params = {'cloudspaceId': int(portforward.network.id), 'publicIp': portforward.network.publicipaddress, 'publicPort': portforward.publicport, 'proto': portforward.protocol}
    self._api_request('/portforwarding/deleteByPort', params)
    return True