from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.python.util import function_utils
def parse_iterator_result(result):
    """Gets features, labels from result."""
    if isinstance(result, (list, tuple)):
        if len(result) != 2:
            raise ValueError('input_fn should return (features, labels) as a len 2 tuple.')
        return (result[0], result[1])
    return (result, None)