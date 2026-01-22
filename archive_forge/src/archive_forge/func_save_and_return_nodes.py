import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def save_and_return_nodes(obj, export_dir, signatures=None, options: save_options.SaveOptions=None, experimental_skip_checkpoint=False):
    """Saves a SavedModel while returning all saved nodes and their paths.

  Please see `tf.saved_model.save` for details.

  Args:
    obj: A trackable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: A function or dictionary of functions to save in the SavedModel
      as signatures.
    options: `tf.saved_model.SaveOptions` object for configuring save options.
    experimental_skip_checkpoint: If set to `True`, the checkpoint will not be
      written.

  Returns:
    A tuple of (a list of saved nodes in the order they are serialized to the
      `SavedObjectGraph`, dictionary mapping nodes to one possible path from
      the root node to the key node)
  """
    options = options or save_options.SaveOptions()
    saved_model = saved_model_pb2.SavedModel()
    meta_graph_def = saved_model.meta_graphs.add()
    _, exported_graph, object_saver, asset_info, saved_nodes, node_paths = _build_meta_graph(obj, signatures, options, meta_graph_def)
    saved_model.saved_model_schema_version = constants.SAVED_MODEL_SCHEMA_VERSION
    if not experimental_skip_checkpoint:
        path_helpers.get_or_create_variables_dir(export_dir)
        ckpt_options = checkpoint_options.CheckpointOptions(experimental_io_device=options.experimental_io_device)
        object_saver.save(path_helpers.get_variables_path(export_dir), options=ckpt_options)
    builder_impl.copy_assets_to_destination_dir(asset_info.asset_filename_map, export_dir)
    if context.executing_eagerly():
        try:
            context.async_wait()
        except errors.NotFoundError as err:
            raise FileNotFoundError(f"{err}\n You may be trying to save on a different device from the computational device. Consider setting the `experimental_io_device` option in `tf.saved_model.SaveOptions` to the io_device such as '/job:localhost'.") from err
    pywrap_saved_model.Save(export_dir)
    if options.experimental_image_format:
        prefix = file_io.join(compat.as_str(export_dir), 'saved_model')
        proto_splitter.SavedModelSplitter(saved_model).write(prefix)
    else:
        path = file_io.join(compat.as_str(export_dir), compat.as_str(constants.SAVED_MODEL_FILENAME_PB))
        file_io.atomic_write_string_to_file(path, saved_model.SerializeToString(deterministic=True))
        fingerprinting_utils.write_fingerprint(export_dir)
    if options.save_debug_info:
        _export_debug_info(exported_graph, export_dir)
    metrics.SetWritePath(saved_model_path=str(export_dir))
    ops.dismantle_graph(exported_graph)
    return (saved_nodes, node_paths)