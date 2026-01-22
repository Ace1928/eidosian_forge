import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
def wrap_f(f):

    def decorator(self, *args, **kwargs):
        """Warms up, gets object counts, runs the test, checks for new objects."""
        with context.eager_mode():
            gc.disable()
            test_errors = None
            test_skipped = None
            if hasattr(self._outcome, 'errors'):
                test_errors = self._outcome.errors
                test_skipped = self._outcome.skipped
            else:
                test_errors = self._outcome.result.errors
                test_skipped = self._outcome.result.skipped
            for _ in range(warmup_iters):
                f(self, *args, **kwargs)
            self.doCleanups()
            obj_count_by_type = _get_object_count_by_type()
            gc.collect()
            registered_function_names = context.context().list_function_names()
            obj_count_by_type = _get_object_count_by_type(exclude=gc.get_referents(test_errors, test_skipped))
            if ops.has_default_graph():
                collection_sizes_before = {collection: len(ops.get_collection(collection)) for collection in ops.get_default_graph().collections}
            for _ in range(3):
                f(self, *args, **kwargs)
            self.doCleanups()
            if ops.has_default_graph():
                for collection_key in ops.get_default_graph().collections:
                    collection = ops.get_collection(collection_key)
                    size_before = collection_sizes_before.get(collection_key, 0)
                    if len(collection) > size_before:
                        raise AssertionError('Collection %s increased in size from %d to %d (current items %s).' % (collection_key, size_before, len(collection), collection))
                    del collection
                    del collection_key
                    del size_before
                del collection_sizes_before
            gc.collect()
            obj_count_by_type = _get_object_count_by_type(exclude=gc.get_referents(test_errors, test_skipped)) - obj_count_by_type
            leftover_functions = context.context().list_function_names() - registered_function_names
            assert not leftover_functions, 'The following functions were newly created: %s' % leftover_functions
            assert not obj_count_by_type, 'The following objects were newly created: %s' % str(obj_count_by_type)
            gc.enable()
    return decorator