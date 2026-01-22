import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@DeveloperAPI
@client_mode_wrap
def placement_group_table(placement_group: PlacementGroup=None) -> dict:
    """Get the state of the placement group from GCS.

    Args:
        placement_group: placement group to see
            states.
    """
    worker = ray._private.worker.global_worker
    worker.check_connected()
    placement_group_id = placement_group.id if placement_group is not None else None
    return ray._private.state.state.placement_group_table(placement_group_id)