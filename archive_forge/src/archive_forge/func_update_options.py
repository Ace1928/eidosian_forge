import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def update_options(original_options: Dict[str, Any], new_options: Dict[str, Any]) -> Dict[str, Any]:
    """Update original options with new options and return.
    The returned updated options contain shallow copy of original options.
    """
    updated_options = {**original_options, **new_options}
    if original_options.get('_metadata') is not None and new_options.get('_metadata') is not None:
        metadata = original_options['_metadata'].copy()
        for namespace, config in new_options['_metadata'].items():
            metadata[namespace] = {**metadata.get(namespace, {}), **config}
        updated_options['_metadata'] = metadata
    return updated_options