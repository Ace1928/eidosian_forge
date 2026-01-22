import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def output_writer_non_streaming(i: PartitionID, shuffle_inputs: List[Any]) -> OutType:
    total = 0
    for arr in shuffle_inputs:
        total += arr.size * arr.itemsize
    return total