import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
@property
def topology(self) -> Topology:
    """A `Topology` that describes the TPU topology."""
    return self._topology