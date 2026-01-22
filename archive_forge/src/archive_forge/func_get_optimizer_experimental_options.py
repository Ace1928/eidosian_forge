import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def get_optimizer_experimental_options(self):
    """Get experimental options for the optimizer.

    Returns:
      Dictionary of current option values
    """
    rewrite_options = self.config.graph_options.rewrite_options
    options = {}

    def rewriter_toggle(option):
        attr = getattr(rewrite_options, option)
        if attr != 0:
            options[option] = attr == rewriter_config_pb2.RewriterConfig.ON

    def rewriter_bool(option):
        options[option] = getattr(rewrite_options, option)
    rewriter_toggle('layout_optimizer')
    rewriter_toggle('constant_folding')
    rewriter_toggle('shape_optimization')
    rewriter_toggle('remapping')
    rewriter_toggle('arithmetic_optimization')
    rewriter_toggle('dependency_optimization')
    rewriter_toggle('loop_optimization')
    rewriter_toggle('function_optimization')
    rewriter_toggle('debug_stripper')
    rewriter_bool('disable_model_pruning')
    rewriter_toggle('scoped_allocator_optimization')
    rewriter_toggle('pin_to_host_optimization')
    rewriter_toggle('implementation_selector')
    rewriter_toggle('auto_mixed_precision')
    rewriter_toggle('use_plugin_optimizers')
    rewriter_bool('disable_meta_optimizer')
    rewriter_toggle('auto_mixed_precision_onednn_bfloat16')
    rewriter_toggle('auto_mixed_precision_mkl')
    if rewrite_options.min_graph_nodes != 0:
        options['min_graph_nodes'] = rewrite_options.min_graph_nodes
    return options