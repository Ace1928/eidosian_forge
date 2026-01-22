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
def worker_replica_patch(group_index: str, target_replicas: int):
    path = f'/spec/workerGroupSpecs/{group_index}/replicas'
    value = target_replicas
    return replace_patch(path, value)