import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util
def when_empty():
    """When input data is empty or bucket_count is zero.

            1. If bucket_count is specified as zero, an empty tensor of shape
              (0, 3) will be returned.
            2. If the input data is empty, a tensor of shape (bucket_count, 3)
              of all zero values will be returned.
            """
    return tf.zeros((bucket_count, 3), dtype=tf.float64)