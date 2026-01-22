import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import ray
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
from ray.util.placement_group import PlacementGroup, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
Resource manager using placement groups as the resource backend.

    This manager will use placement groups to fulfill resource requests. Requesting
    a resource will schedule the placement group. Acquiring a resource will
    return a ``PlacementGroupAcquiredResources`` that can be used to schedule
    Ray tasks and actors on the placement group. Freeing an acquired resource
    will destroy the associated placement group.

    Ray core does not emit events when resources are available. Instead, the
    scheduling state has to be periodically updated.

    Per default, placement group scheduling state is refreshed every time when
    resource state is inquired, but not more often than once every ``update_interval_s``
    seconds. Alternatively, staging futures can be retrieved (and awaited) with
    ``get_resource_futures()`` and state update can be force with ``update_state()``.

    Args:
        update_interval_s: Minimum interval in seconds between updating scheduling
            state of placement groups.

    