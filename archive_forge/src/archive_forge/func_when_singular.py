import numpy as np
from tensorboard.plugins.histogram import metadata
from tensorboard.plugins.histogram import summary_v2
def when_singular():
    center = min_
    bucket_starts = tf.stack([center - 0.5])
    bucket_ends = tf.stack([center + 0.5])
    bucket_counts = tf.stack([tf.cast(tf.size(input=data), tf.float64)])
    return tf.transpose(a=tf.stack([bucket_starts, bucket_ends, bucket_counts]))