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
def assert_meta_graph_protos_equal(tester, a, b):
    """Compares MetaGraphDefs `a` and `b` in unit test class `tester`."""
    tester.assertEqual(set(a.collection_def), set(b.collection_def))
    collection_keys = a.collection_def.keys()
    for k in collection_keys:
        a_value = a.collection_def[k]
        b_value = b.collection_def[k]
        proto_type = ops.get_collection_proto_type(k)
        if proto_type:
            a_proto = proto_type()
            b_proto = proto_type()
            tester.assertEqual(len(a_value.bytes_list.value), len(b_value.bytes_list.value))
            for a_value_item, b_value_item in zip(a_value.bytes_list.value, b_value.bytes_list.value):
                a_proto.ParseFromString(a_value_item)
                b_proto.ParseFromString(b_value_item)
                tester.assertProtoEquals(a_proto, b_proto)
        else:
            tester.assertEquals(a_value, b_value)
    a.ClearField('collection_def')
    b.ClearField('collection_def')
    assert_equal_graph_def(a.graph_def, b.graph_def, checkpoint_v2=True)
    tester.assertProtoEquals(a.graph_def.versions, b.graph_def.versions)
    a.ClearField('graph_def')
    b.ClearField('graph_def')
    tester.assertProtoEquals(a, b)