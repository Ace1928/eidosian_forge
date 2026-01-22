import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
def get_device_locations(mesh: layout_lib.Mesh, client_id: Optional[int]=None) -> List[Dict[str, int]]:
    """Returns the device locations of all TPU cores local to the given client.

  A device location is a dictionary from dimension names to indices on those
  dimensions. For example, for a 2x2 mesh ('x', 'y'), this function returns a
  permutation of this list:

    [{'x': 0, 'y': 0},
     {'x': 0, 'y': 1},
     {'x': 1, 'y': 0},
     {'x': 1, 'y': 1}].

  Note that device IDs and device locations are equivalent. The former is a
  linearization of the latter along mesh dimensions.

  Args:
    mesh: A TPU mesh.
    client_id: Optional; A DTensor client ID. If empty, query this client.
  """
    if mesh.device_type() != _TPU_DEVICE_TYPE:
        raise ValueError('The mesh must be a TPU mesh')
    if client_id is None or client_id == config.client_id():
        return mesh.local_device_locations()
    raise NotImplementedError("Looking up other clients' device locations is not supported")