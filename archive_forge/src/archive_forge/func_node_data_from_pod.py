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
def node_data_from_pod(pod: Dict[str, Any]) -> NodeData:
    """Converts a Ray pod extracted from K8s into Ray NodeData.
    NodeData is processed by BatchingNodeProvider.
    """
    kind, type = kind_and_type(pod)
    status = status_tag(pod)
    ip = pod_ip(pod)
    return NodeData(kind=kind, type=type, status=status, ip=ip)