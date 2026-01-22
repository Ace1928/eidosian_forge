import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_portforward(self, network, node, publicport, privateport, protocol='tcp'):
    params = {'cloudspaceId': int(network.id), 'machineId': int(node.id), 'localPort': privateport, 'publicPort': publicport, 'publicIp': network.publicipaddress, 'protocol': protocol}
    self._api_request('/portforwarding/create', params)
    return self._to_port_forward(params, network)