import threading
import unittest
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.optimizers.legacy import gradient_descent
from tensorflow.python.distribute.cluster_resolver import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import (
def make_parameter_server_cluster(num_workers, num_ps):
    cluster_def = create_in_process_cluster(num_workers=num_workers, num_ps=num_ps, rpc_layer='grpc')
    return SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer='grpc')