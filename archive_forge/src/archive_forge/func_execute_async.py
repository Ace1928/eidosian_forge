import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
def execute_async(self, func: Callable[..., T], *args, **kwargs) -> List[ObjectRef]:
    """Execute ``func`` on each worker and return the futures.

        Args:
            func: A function to call on each worker.
            args, kwargs: Passed directly into func.

        Returns:
            (List[ObjectRef]) A list of ``ObjectRef`` representing the
                output of ``func`` from each worker. The order is the same
                as ``self.workers``.

        """
    if len(self.workers) <= 0:
        raise RuntimeError('There are no active workers. This worker group has most likely been shut down. Pleasecreate a new WorkerGroup or restart this one.')
    return [w.actor._RayTrainWorker__execute.options(name=f'_RayTrainWorker__execute.{func.__name__}').remote(func, *args, **kwargs) for w in self.workers]