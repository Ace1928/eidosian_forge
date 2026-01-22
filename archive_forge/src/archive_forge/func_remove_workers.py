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
def remove_workers(self, worker_indexes: List[int]):
    """Removes the workers with the specified indexes.

        The removed workers will go out of scope and their actor processes
        will be terminated.

        Args:
            worker_indexes (List[int]): The indexes of the workers to remove.
        """
    new_workers = []
    for i in range(len(self.workers)):
        if i not in worker_indexes:
            new_workers.append(self.workers[i])
    self.workers = new_workers