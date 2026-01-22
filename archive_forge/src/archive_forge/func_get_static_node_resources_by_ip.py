import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def get_static_node_resources_by_ip(self) -> Dict[NodeIP, ResourceDict]:
    """Return a dict of node resources for every node ip.

        Example:
            >>> from ray.autoscaler._private.load_metrics import LoadMetrics
            >>> metrics = LoadMetrics(...)  # doctest: +SKIP
            >>> metrics.get_static_node_resources_by_ip()  # doctest: +SKIP
            {127.0.0.1: {"CPU": 1}, 127.0.0.2: {"CPU": 4, "GPU": 8}}
        """
    return self.static_resources_by_ip