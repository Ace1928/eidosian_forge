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
def safe_to_scale(self) -> bool:
    """Returns False iff non_terminated_nodes contains any pods in the RayCluster's
        workersToDelete lists.

        Explanation:
        If there are any workersToDelete which are non-terminated,
        we should wait for the operator to do its job and delete those
        pods. Therefore, we back off the autoscaler update.

        If, on the other hand, all of the workersToDelete have already been cleaned up,
        then we patch away the workersToDelete lists and return True.
        In the future, we may consider having the operator clean up workersToDelete
        on it own:
        https://github.com/ray-project/kuberay/issues/733

        Note (Dmitri):
        It is stylistically bad that this function has a side effect.
        """
    node_set = set(self.node_data_dict.keys())
    worker_groups = self._raycluster['spec'].get('workerGroupSpecs', [])
    non_empty_worker_group_indices = []
    for group_index, worker_group in enumerate(worker_groups):
        workersToDelete = worker_group.get('scaleStrategy', {}).get('workersToDelete', [])
        if workersToDelete:
            non_empty_worker_group_indices.append(group_index)
        for worker in workersToDelete:
            if worker in node_set:
                logger.warning(f'Waiting for operator to remove worker {worker}.')
                return False
    patch_payload = []
    for group_index in non_empty_worker_group_indices:
        patch = worker_delete_patch(group_index, workers_to_delete=[])
        patch_payload.append(patch)
    if patch_payload:
        logger.info('Cleaning up workers to delete.')
        logger.info(f'Submitting patch {patch_payload}.')
        self._submit_raycluster_patch(patch_payload)
    return True