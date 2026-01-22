import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def read_element(dataset, index):
    shuffled_index = stateless_random_ops.index_shuffle(index, seeds, result['num_elements'] - 1)
    if 'thresholds' in result and 'offsets' in result:
        shuffled_index = _adjust_index(shuffled_index, result['thresholds'], result['offsets'])
    return random_access.at(dataset, shuffled_index)