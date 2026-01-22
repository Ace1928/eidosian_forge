import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
def batch_delete_instances(self, instance_ids: List[str], expected_storage_version: Optional[int]=None) -> StoreStatus:
    """Delete instances from the storage. If the expected_version is
        specified, the update will fail if the current storage version does not
        match the expected version.

        Args:
            to_delete: A list of instances to be deleted.
            expected_version: The expected storage version.

        Returns:
            StoreStatus: A tuple of (success, storage_version).
        """
    version = self._storage.get_version()
    if expected_storage_version and expected_storage_version != version:
        return StoreStatus(False, version)
    result = self._storage.batch_update(self._table_name, {}, instance_ids, expected_storage_version)
    if result[0]:
        for subscriber in self._status_change_subscribers:
            subscriber.notify([InstanceUpdateEvent(instance_id=instance_id, new_status=Instance.GARBAGE_COLLECTED, new_ray_status=Instance.RAY_STATUS_UNKOWN) for instance_id in instance_ids])
    return result