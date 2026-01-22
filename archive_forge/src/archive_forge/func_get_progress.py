import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def get_progress(self):
    if self.map_refs:
        ready, self.map_refs = ray.wait(self.map_refs, timeout=1, num_returns=len(self.map_refs), fetch_local=False)
        self.num_map += len(ready)
    elif self.reduce_refs:
        ready, self.reduce_refs = ray.wait(self.reduce_refs, timeout=1, num_returns=len(self.reduce_refs), fetch_local=False)
        self.num_reduce += len(ready)
    return (self.num_map, self.num_reduce)