from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
Canonicalizes either relative or absolute times, with error checking.