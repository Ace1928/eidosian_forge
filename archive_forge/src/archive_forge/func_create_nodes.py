import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.tags import TAG_RAY_USER_NODE_TYPE
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance
def create_nodes(self, instance_type_name: str, count: int) -> List[Instance]:
    created_nodes = self._node_launcher.launch_node(self._config.get_raw_config_mutable(), count, instance_type_name)
    if created_nodes:
        return [self._get_instance(cloud_instance_id) for cloud_instance_id in created_nodes.keys()]
    return []