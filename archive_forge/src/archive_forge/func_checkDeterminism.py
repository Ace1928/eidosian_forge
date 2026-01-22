import os
import random
import re
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
def checkDeterminism(self, dataset_fn, expect_determinism, expected_elements):
    """Tests whether a dataset produces its elements deterministically.

    `dataset_fn` takes a delay_ms argument, which tells it how long to delay
    production of the first dataset element. This gives us a way to trigger
    out-of-order production of dataset elements.

    Args:
      dataset_fn: A function taking a delay_ms argument.
      expect_determinism: Whether to expect deterministic ordering.
      expected_elements: The elements expected to be produced by the dataset,
        assuming the dataset produces elements in deterministic order.
    """
    if expect_determinism:
        dataset = dataset_fn(100)
        actual = self.getDatasetOutput(dataset)
        self.assertAllEqual(expected_elements, actual)
        return
    for delay_ms in [10, 100, 1000, 20000, 100000]:
        dataset = dataset_fn(delay_ms)
        actual = self.getDatasetOutput(dataset)
        self.assertCountEqual(expected_elements, actual)
        for i in range(len(actual)):
            if actual[i] != expected_elements[i]:
                return
    self.fail('Failed to observe nondeterministic ordering')