import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
@classmethod
def assign_device(cls, core):
    """Returns an AssignDevice sharding attribute.

    This causes an op to be computed in its entirety only on one core in
    the XLA device.
    Args:
      core: The core to assign this Op to.
    """
    return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MAXIMAL, tile_assignment_dimensions=[1], tile_assignment_devices=[core]))