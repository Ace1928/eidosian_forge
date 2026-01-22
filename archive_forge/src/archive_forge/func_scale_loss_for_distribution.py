from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
def scale_loss_for_distribution(loss_value):
    """Scales and returns the given loss value by the number of replicas."""
    num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= 1.0 / num_replicas
    return loss_value