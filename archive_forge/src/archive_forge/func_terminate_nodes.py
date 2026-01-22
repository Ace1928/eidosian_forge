import json
import logging
from http.client import RemoteDisconnected
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME
def terminate_nodes(self, node_ids):
    request = {'type': 'terminate_nodes', 'args': (node_ids,)}
    self._get_http_response(request)