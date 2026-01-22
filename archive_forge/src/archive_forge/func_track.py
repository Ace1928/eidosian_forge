from typing import List, Set, Tuple
from ray.autoscaler._private import constants
def track(self, node_id: str, ip: str, node_type: str):
    """
        Begin to track a new node.

        Args:
            node_id: The node id.
            ip: The node ip address.
            node_type: The node type.
        """
    if node_id not in self.node_mapping:
        self._add_node_mapping(node_id, (ip, node_type))