import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def get_node_data(self) -> Dict[NodeID, NodeData]:
    """Queries K8s for pods in the RayCluster. Converts that pod data into a
        map of pod name to Ray NodeData, as required by BatchingNodeProvider.
        """
    self._raycluster = self._get(f'rayclusters/{self.cluster_name}')
    resource_version = self._get_pods_resource_version()
    if resource_version:
        logger.info(f'Listing pods for RayCluster {self.cluster_name} in namespace {self.namespace} at pods resource version >= {resource_version}.')
    label_selector = requests.utils.quote(f'ray.io/cluster={self.cluster_name}')
    resource_path = f'pods?labelSelector={label_selector}'
    if resource_version:
        resource_path += f'&resourceVersion={resource_version}' + '&resourceVersionMatch=NotOlderThan'
    pod_list = self._get(resource_path)
    fetched_resource_version = pod_list['metadata']['resourceVersion']
    logger.info(f'Fetched pod data at resource version {fetched_resource_version}.')
    node_data_dict = {}
    for pod in pod_list['items']:
        if 'deletionTimestamp' in pod['metadata']:
            continue
        pod_name = pod['metadata']['name']
        node_data_dict[pod_name] = node_data_from_pod(pod)
    return node_data_dict