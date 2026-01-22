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
@client_mode_wrap
def remove_placement_group(placement_group: PlacementGroup) -> None:
    """Asynchronously remove placement group.

    Args:
        placement_group: The placement group to delete.
    """
    assert placement_group is not None
    worker = ray._private.worker.global_worker
    worker.check_connected()
    worker.core_worker.remove_placement_group(placement_group.id)