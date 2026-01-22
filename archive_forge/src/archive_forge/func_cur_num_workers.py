import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def cur_num_workers(self):
    """Returns dict mapping node type to the number of nodes of that type."""
    return self._cur_num_workers(self.node_data_dict)