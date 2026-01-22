import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
def run_fc(data, fc_fn, batch_size, num_runs, steps_per_repeat=100):
    """Benchmark a Feature Column."""
    ds = tf.data.Dataset.from_tensor_slices(data).repeat().prefetch(tf.data.AUTOTUNE).batch(batch_size).cache()
    ds_iter = ds.__iter__()
    fc_fn(next(ds_iter))
    fc_starts = []
    fc_ends = []
    for _ in range(num_runs):
        fc_starts.append(time.time())
        for _ in range(steps_per_repeat):
            _ = fc_fn(next(ds_iter))
        fc_ends.append(time.time())
    avg_per_step_time = (np.array(fc_ends) - np.array(fc_starts)) / steps_per_repeat
    avg_time = np.mean(avg_per_step_time)
    return avg_time