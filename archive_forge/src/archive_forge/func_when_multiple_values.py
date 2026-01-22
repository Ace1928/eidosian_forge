import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util
def when_multiple_values():
    """When input data contains multiple values."""
    bucket_width = range_ / tf.cast(bucket_count, tf.float64)
    offsets = data - min_
    bucket_indices = tf.cast(tf.floor(offsets / bucket_width), dtype=tf.int32)
    clamped_indices = tf.minimum(bucket_indices, bucket_count - 1)
    one_hots = tf.one_hot(clamped_indices, depth=bucket_count, dtype=tf.float64)
    bucket_counts = tf.cast(tf.reduce_sum(input_tensor=one_hots, axis=0), dtype=tf.float64)
    edges = tf.linspace(min_, max_, bucket_count + 1)
    edges = tf.concat([edges[:-1], [max_]], 0)
    left_edges = edges[:-1]
    right_edges = edges[1:]
    return tf.transpose(a=tf.stack([left_edges, right_edges, bucket_counts]))