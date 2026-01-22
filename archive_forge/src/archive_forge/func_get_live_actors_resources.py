import logging
import random
import time
import uuid
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import ray
from ray.air.execution._internal.event_manager import RayEventManager
from ray.air.execution.resources import (
from ray.air.execution._internal.tracked_actor import TrackedActor
from ray.air.execution._internal.tracked_actor_task import TrackedActorTask
from ray.exceptions import RayTaskError, RayActorError
def get_live_actors_resources(self):
    if self._live_resource_cache:
        return self._live_resource_cache
    counter = Counter()
    for _, acq in self._live_actors_to_ray_actors_resources.values():
        for bdl in acq.resource_request.bundles:
            counter.update(bdl)
    self._live_resource_cache = dict(counter)
    return self._live_resource_cache