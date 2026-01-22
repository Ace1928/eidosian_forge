import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def pick_actor(self, locality_hint: Optional[RefBundle]=None) -> Optional[ray.actor.ActorHandle]:
    """Picks an actor for task submission based on busyness and locality.

        None will be returned if all actors are either at capacity (according to
        max_tasks_in_flight) or are still pending.

        Args:
            locality_hint: Try to pick an actor that is local for this bundle.
        """
    if not self._num_tasks_in_flight:
        return None
    if locality_hint:
        preferred_loc = self._get_location(locality_hint)
    else:
        preferred_loc = None

    def penalty_key(actor):
        """Returns the key that should be minimized for the best actor.

            We prioritize valid actors, those with argument locality, and those that
            are not busy, in that order.
            """
        busyness = self._num_tasks_in_flight[actor]
        invalid = busyness >= self._max_tasks_in_flight
        requires_remote_fetch = self._actor_locations[actor] != preferred_loc
        return (invalid, requires_remote_fetch, busyness)
    actor = min(self._num_tasks_in_flight.keys(), key=penalty_key)
    if self._num_tasks_in_flight[actor] >= self._max_tasks_in_flight:
        return None
    if locality_hint:
        if self._actor_locations[actor] == preferred_loc:
            self._locality_hits += 1
        else:
            self._locality_misses += 1
    self._num_tasks_in_flight[actor] += 1
    return actor