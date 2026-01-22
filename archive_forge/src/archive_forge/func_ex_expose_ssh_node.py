import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_expose_ssh_node(self, node):
    """
        Create portforward for ssh purposed

        :param node: Node to expose ssh for
        :type  node: ``Node``

        :rtype: ``int``
        """
    network = node.extra['network']
    ports = self._find_ssh_ports(network, node)
    if ports['node']:
        return ports['node']
    usedports = ports['network']
    sshport = 2200
    endport = 3000
    while sshport < endport:
        while sshport in usedports:
            sshport += 1
        try:
            network.create_portforward(node, sshport, 22)
            node.extra['ssh_port'] = sshport
            node.extra['ssh_ip'] = network.publicipaddress
            break
        except BaseHTTPError as e:
            if e.code == 409:
                usedports.append(sshport)
            raise
    else:
        raise G8ProvisionError('Failed to create portforward')
    return sshport