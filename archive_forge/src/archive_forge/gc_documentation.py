from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
A filter function that keeps exactly one out of every n paths.