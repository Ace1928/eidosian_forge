import itertools
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import hashing
def run_dataset_implementation(self, batch_size):
    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
        ds = tf.data.Dataset.from_generator(word_gen, tf.string, tf.TensorShape([]))
        ds = ds.shuffle(batch_size * 100)
        ds = ds.batch(batch_size)
        num_batches = 5
        ds = ds.take(num_batches)
        ds = ds.prefetch(num_batches)
        starts.append(time.time())
        for i in ds:
            _ = tf.strings.to_hash_bucket(i, num_buckets=2)
        ends.append(time.time())
    avg_time = np.mean(np.array(ends) - np.array(starts)) / num_batches
    return avg_time