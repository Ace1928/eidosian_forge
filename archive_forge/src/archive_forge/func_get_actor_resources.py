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
def get_actor_resources(self, tracked_actor: TrackedActor) -> Optional[AcquiredResources]:
    """Returns the acquired resources of an actor that has been started.

        This will return ``None`` if the actor has not been started, yet.

        Args:
            tracked_actor: Tracked actor object.
        """
    if not self.is_actor_started(tracked_actor):
        return None
    return self._live_actors_to_ray_actors_resources[tracked_actor][1]