import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def tpu_device_ordinal_at_coordinates(self, device_coordinates):
    """Returns the TensorFlow device number at `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow device number within the task corresponding to
      attached to the device with those physical coordinates.
    """
    return self._topology_devices[tuple(device_coordinates)]