from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def get_default_session_config():
    """Returns tf.ConfigProto instance."""
    rewrite_opts = rewriter_config_pb2.RewriterConfig(meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE)
    graph_opts = tf.compat.v1.GraphOptions(rewrite_options=rewrite_opts)
    return tf.compat.v1.ConfigProto(allow_soft_placement=True, graph_options=graph_opts)