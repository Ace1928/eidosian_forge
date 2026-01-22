import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def calibrate(self, fetch_names, num_runs, feed_dict_fn=None, input_map_fn=None):
    """Run the calibration and return the calibrated GraphDef.

    Args:
      fetch_names: a list of output tensor name to fetch during calibration.
      num_runs: number of runs of the graph during calibration.
      feed_dict_fn: a function that returns a dictionary mapping input names (as
        strings) in the GraphDef to be calibrated to values (e.g. Python list,
        numpy arrays, etc). One and only one of `feed_dict_fn` and
        `input_map_fn` should be specified.
      input_map_fn: a function that returns a dictionary mapping input names (as
        strings) in the GraphDef to be calibrated to Tensor objects. The values
        of the named input tensors in the GraphDef to be calibrated will be
        re-mapped to the respective `Tensor` values during calibration. One and
        only one of `feed_dict_fn` and `input_map_fn` should be specified.

    Raises:
      ValueError: if the input combination is invalid.
      RuntimeError: if this method is called in eager mode.

    Returns:
      The GraphDef after the calibration.
    """
    assert self._converted
    assert self._need_calibration
    assert not self._calibration_data_collected
    if feed_dict_fn and input_map_fn or (not feed_dict_fn and (not input_map_fn)):
        raise ValueError('Should specify one and only one of feed_dict_fn and input_map_fn.')
    if input_map_fn:
        for k, v in input_map_fn().items():
            if not isinstance(k, str):
                raise ValueError('Keys of input_map_fn must be of type str')
            if not isinstance(v, tensor.Tensor):
                raise ValueError('Values of input_map_fn must be of type tf.Tensor')
    self._calibration_graph = ops.Graph()
    with self._calibration_graph.as_default():
        fetches = importer.import_graph_def(self._converted_graph_def, input_map=input_map_fn() if input_map_fn else None, return_elements=fetch_names, name='')
    calibrate_rewriter_cfg = rewriter_config_pb2.RewriterConfig()
    if self._test_only_disable_non_trt_optimizers:
        trt_utils.disable_non_trt_optimizers_in_rewriter_config(calibrate_rewriter_cfg)
    calibrate_config = config_pb2.ConfigProto(allow_soft_placement=True, graph_options=config_pb2.GraphOptions(rewrite_options=calibrate_rewriter_cfg))
    with session.Session(graph=self._calibration_graph, config=calibrate_config) as calibration_sess:
        for _ in range(num_runs):
            calibration_sess.run(fetches, feed_dict=feed_dict_fn() if feed_dict_fn else None)
        device_to_get_resource_op_map = {}
        with self._calibration_graph.as_default():
            resource_name_input = array_ops.placeholder(dtypes.string)
            for node in self._converted_graph_def.node:
                if node.op == _TRT_ENGINE_OP_NAME:
                    if node.device not in device_to_get_resource_op_map:
                        with self._calibration_graph.device(node.device):
                            serialized_resources_output = gen_trt_ops.get_calibration_data_op(resource_name_input)
                        device_to_get_resource_op_map[node.device] = serialized_resources_output
                    calibration_result = calibration_sess.run(device_to_get_resource_op_map[node.device], feed_dict={resource_name_input: _get_canonical_engine_name(node.name)})
                    node.attr['calibration_data'].s = calibration_result
        self._calibration_data_collected = True
    return self._converted_graph_def