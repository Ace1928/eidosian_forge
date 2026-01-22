import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def validate_actor_options(options: Dict[str, Any], in_options: bool):
    """Options check for Ray actors.

    Args:
        options: Options for Ray actors.
        in_options: If True, we are checking the options under the context of
            ".options()".
    """
    for k, v in options.items():
        if k not in actor_options:
            raise ValueError(f'Invalid option keyword {k} for actors. Valid ones are {list(actor_options)}.')
        actor_options[k].validate(k, v)
    if in_options and 'concurrency_groups' in options:
        raise ValueError("Setting 'concurrency_groups' is not supported in '.options()'.")
    if options.get('get_if_exists') and (not options.get('name')):
        raise ValueError('The actor name must be specified to use `get_if_exists`.')
    if 'object_store_memory' in options:
        warnings.warn(f"Setting 'object_store_memory' for actors is deprecated since it doesn't actually reserve the required object store memory. Use object spilling that's enabled by default (https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/objects/object-spilling.html) instead to bypass the object store memory size limitation.", DeprecationWarning, stacklevel=1)
    _check_deprecate_placement_group(options)