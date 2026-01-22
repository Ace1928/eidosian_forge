from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import ray
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_GCS_RPC_TIMEOUT_S
def get_alive_nodes(self) -> List[Tuple[str, str]]:
    """Get IDs and IPs for all live nodes in the cluster.

        Returns a list of (node_id: str, ip_address: str). The node_id can be
        passed into the Ray SchedulingPolicy API.
        """
    return self._cached_alive_nodes