from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import time
import numpy as np
import six
import tensorflow as tf
def check_positive_integer(value, name):
    """Checks whether `value` is a positive integer."""
    if not isinstance(value, (six.integer_types, np.integer)):
        raise TypeError('{} must be int, got {}'.format(name, type(value)))
    if value <= 0:
        raise ValueError('{} must be positive, got {}'.format(name, value))