import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
def tpu_device(self, replica: int=0, logical_core: int=0, job: Optional[str]=None) -> str:
    """Returns the name of the TPU device assigned to a logical core."""
    coordinates = self.coordinates(replica, logical_core)
    return self._topology.tpu_device_name_at_coordinates(coordinates, job=job)