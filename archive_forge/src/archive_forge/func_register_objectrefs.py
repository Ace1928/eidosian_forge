import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
def register_objectrefs(self, map_refs, reduce_refs):
    self.map_refs = map_refs
    self.reduce_refs = reduce_refs