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
def get_lowest_gpu_id(worker) -> int:
    gpu_ids = worker.metadata.resource_ids.get('GPU', [])
    if not gpu_ids:
        return 0
    try:
        return min((int(gpu_id) for gpu_id in gpu_ids))
    except ValueError:
        return min(gpu_ids)