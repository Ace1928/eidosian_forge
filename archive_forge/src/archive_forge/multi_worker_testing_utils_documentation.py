import threading
import unittest
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.optimizers.legacy import gradient_descent
from tensorflow.python.distribute.cluster_resolver import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import (
Create an in-process cluster that consists of only standard server.