import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
def offset_to_shard(self):
    """Mapping from offset in a flattened list to shard index."""
    unravel_index = self.mesh.unravel_index()
    locations = [None] * self.mesh.size
    for offset, mesh_loc in unravel_index.items():
        loc = []
        for dim_sharding in self.sharding_specs:
            if dim_sharding == UNSHARDED:
                loc.append(0)
            else:
                loc.append(mesh_loc[dim_sharding])
        locations[offset] = tuple(loc)
    return locations