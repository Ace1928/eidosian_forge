import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def scan_random_index(state, index):
    if tf.reduce_all(state == -1):
        state = tf.random.uniform(shape=(3,), maxval=2 ** 62, dtype=tf.int64)
    shuffled_index = random_index_shuffle(index=index, seed=state, max_index=len(dataset) - 1)
    return (state, shuffled_index)