import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
@classmethod
def from_schedule_request(cls, req: SchedulingRequest) -> 'ResourceDemandScheduler.ScheduleContext':
    """
            Create a schedule context from a schedule request.
            It will populate the context with the existing nodes and the available node
            types from the config.

            Args:
                req: The scheduling request. The caller should make sure the
                    request is valid.
            """
    nodes = []
    for node in req.current_nodes:
        nodes.append(SchedulingNode(node_type=node.ray_node_type_name, total_resources=dict(node.total_resources), available_resources=dict(node.available_resources), labels=dict(node.dynamic_labels), status=SchedulingNodeStatus.RUNNING))
    cluster_config = req.cluster_config
    for instance in req.current_instances:
        if not is_pending(instance):
            continue
        node_config = cluster_config.node_type_configs[instance.ray_node_type_name]
        nodes.append(SchedulingNode.from_node_config(node_config, status=SchedulingNodeStatus.PENDING))
    node_type_available = cls._compute_available_node_types(nodes, req.cluster_config)
    return cls(nodes=nodes, node_type_available=node_type_available, cluster_config=req.cluster_config)