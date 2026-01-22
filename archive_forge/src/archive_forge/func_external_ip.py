import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def external_ip(self, node_id):
    """Returns an external ip if the user has supplied one.
        Otherwise, use the same logic as internal_ip below.

        This can be used to call ray up from outside the network, for example
        if the Ray cluster exists in an AWS VPC and we're interacting with
        the cluster from a laptop (where using an internal_ip will not work).

        Useful for debugging the local node provider with cloud VMs."""
    node_state = self.state.get()[node_id]
    ext_ip = node_state.get('external_ip')
    if ext_ip:
        return ext_ip
    else:
        return socket.gethostbyname(node_id)