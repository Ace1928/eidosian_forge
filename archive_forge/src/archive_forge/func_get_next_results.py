import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def get_next_results(self) -> Optional[List[_TrainingResult]]:
    """Fetches the next ``_TrainingResult`` from each worker.

        Each ``_TrainingResult`` is expected to correspond to the same step from
        each worker (e.g. the same call to ``train.report()``).

        Returns:
            A list of ``_TrainingResult``s or ``None`` if there are no more results
            since the training function has exited on all workers.
        """

    def get_next():
        session = _get_session('get_next_results')
        try:
            result = session.get_next()
        except RuntimeError:
            raise TrainBackendError('`get_next_results` has been called before `start_training`. Please call `start_training` before `get_next_results`.')
        return result
    futures = self.worker_group.execute_async(get_next)
    results = self.get_with_failure_handling(futures)
    if any((r is None for r in results)):
        if not all((r is None for r in results)):
            raise RuntimeError("Some workers returned results while others didn't. Make sure that `session.report()` are called the same number of times on all workers.")
        else:
            return None
    return results