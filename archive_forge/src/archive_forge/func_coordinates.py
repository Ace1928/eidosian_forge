import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
def coordinates(self, replica: int, logical_core: int) -> Tuple:
    """Returns the physical topology coordinates of a logical core."""
    return tuple(self.core_assignment[replica, logical_core, :])