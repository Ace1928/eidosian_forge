import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def get_resource_demand_vector(self, clip=True):
    if clip:
        return self.waiting_bundles[:AUTOSCALER_MAX_RESOURCE_DEMAND_VECTOR_SIZE] + self.infeasible_bundles[:AUTOSCALER_MAX_RESOURCE_DEMAND_VECTOR_SIZE]
    else:
        return self.waiting_bundles + self.infeasible_bundles