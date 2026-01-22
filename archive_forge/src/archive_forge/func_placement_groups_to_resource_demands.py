import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def placement_groups_to_resource_demands(pending_placement_groups: List[PlacementGroupTableData]):
    """Preprocess placement group requests into regular resource demand vectors
    when possible. The policy is:
        * STRICT_PACK - Convert to a single bundle.
        * PACK - Flatten into a resource demand vector.
        * STRICT_SPREAD - Cannot be converted.
        * SPREAD - Flatten into a resource demand vector.

    Args:
        pending_placement_groups (List[PlacementGroupData]): List of
        PlacementGroupLoad's.

    Returns:
        List[ResourceDict]: The placement groups which were converted to a
            resource demand vector.
        List[List[ResourceDict]]: The placement groups which should be strictly
            spread.
    """
    resource_demand_vector = []
    unconverted = []
    for placement_group in pending_placement_groups:
        shapes = [dict(bundle.unit_resources) for bundle in placement_group.bundles]
        if placement_group.strategy == PlacementStrategy.PACK or placement_group.strategy == PlacementStrategy.SPREAD:
            resource_demand_vector.extend(shapes)
        elif placement_group.strategy == PlacementStrategy.STRICT_PACK:
            combined = collections.defaultdict(float)
            for shape in shapes:
                for label, quantity in shape.items():
                    combined[label] += quantity
            resource_demand_vector.append(combined)
        elif placement_group.strategy == PlacementStrategy.STRICT_SPREAD:
            unconverted.append(shapes)
        else:
            logger.error(f'Unknown placement group request type: {placement_group}. Please file a bug report https://github.com/ray-project/ray/issues/new.')
    return (resource_demand_vector, unconverted)