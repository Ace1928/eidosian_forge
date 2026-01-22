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
def status_tag(pod: Dict[str, Any]) -> NodeStatus:
    """Convert pod state to Ray autoscaler node status.

    See the doc string of the class
    batching_node_provider.NodeData for the semantics of node status.
    """
    if 'containerStatuses' not in pod['status'] or not pod['status']['containerStatuses']:
        return 'pending'
    state = pod['status']['containerStatuses'][0]['state']
    if 'pending' in state:
        return 'pending'
    if 'running' in state:
        return STATUS_UP_TO_DATE
    if 'waiting' in state:
        return 'waiting'
    if 'terminated' in state:
        return STATUS_UPDATE_FAILED
    raise ValueError('Unexpected container state.')