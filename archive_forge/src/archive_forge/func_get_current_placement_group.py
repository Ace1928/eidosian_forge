import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@PublicAPI
def get_current_placement_group() -> Optional[PlacementGroup]:
    """Get the current placement group which a task or actor is using.

    It returns None if there's no current placement group for the worker.
    For example, if you call this method in your driver, it returns None
    (because drivers never belong to any placement group).

    Examples:
        .. testcode::

            import ray
            from ray.util.placement_group import get_current_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            @ray.remote
            def f():
                # This returns the placement group the task f belongs to.
                # It means this pg is identical to the pg created below.
                return get_current_placement_group()

            pg = ray.util.placement_group([{"CPU": 2}])
            assert ray.get(f.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg)).remote()) == pg

            # Driver doesn't belong to any placement group,
            # so it returns None.
            assert get_current_placement_group() is None

    Return:
        PlacementGroup: Placement group object.
            None if the current task or actor wasn't
            created with any placement group.
    """
    auto_init_ray()
    if client_mode_should_convert():
        return None
    worker = ray._private.worker.global_worker
    worker.check_connected()
    pg_id = worker.placement_group_id
    if pg_id.is_nil():
        return None
    return PlacementGroup(pg_id)