import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
@classmethod
def manual(cls):
    """Returns a manuall sharding attribute.

    This means the op is manually partitioned by the user and XLA will not
    change the shapes.
    """
    return Sharding(proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MANUAL))