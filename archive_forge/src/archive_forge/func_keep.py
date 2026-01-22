from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def keep(paths):
    l = set(paths)
    r = set(f(paths))
    return sorted(list(l - r))