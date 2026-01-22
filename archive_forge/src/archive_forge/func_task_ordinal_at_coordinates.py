import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def task_ordinal_at_coordinates(self, device_coordinates):
    """Returns the TensorFlow task number attached to `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow task number that contains the TPU device with those
      physical coordinates.
    """
    return self._topology_tasks[tuple(device_coordinates)]