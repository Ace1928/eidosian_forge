import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def frozen_saver(root_trackable):
    """Creates a static `tf.compat.v1.train.Saver` from a trackable object.

  The returned `Saver` saves object-based checkpoints, but these checkpoints
  will no longer reflect structural changes to the object graph, only changes to
  the values of `Variable`s added as dependencies of the root object before
  `freeze` was called.

  `restore` works on the returned `Saver`, but requires that the object graph of
  the checkpoint being loaded exactly matches the object graph when `freeze` was
  called. This is in contrast the object-based restore performed by
  `tf.train.Checkpoint` which attempts a fuzzy matching between a checkpoint's
  object graph and the current Python object graph.

  Args:
    root_trackable: A trackable object to save.

  Returns:
    A saver which saves object-based checkpoints for the object graph frozen at
    the time `frozen_saver` was called.
  """
    named_saveable_objects, registered_savers = save_util_v1.frozen_saveables_and_savers(graph_view_lib.ObjectGraphView(root_trackable))
    return functional_saver.MultiDeviceSaver.from_saveables(named_saveable_objects, registered_savers)