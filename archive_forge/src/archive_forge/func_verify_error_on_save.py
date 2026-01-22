import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest
def verify_error_on_save(self, ds_fn, num_outputs, error, break_point=None, sparse_tensors=False):
    """Attempts to save a non-saveable iterator.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      error: Declared error when trying to save iterator.
      break_point: Break point. Optional. Defaults to num_outputs/2.
      sparse_tensors: Whether dataset is built from SparseTensor(s).

    Raises:
      AssertionError if any test fails.
    """
    break_point = num_outputs // 2 if not break_point else break_point
    if context.executing_eagerly():
        iterator = iter(ds_fn())
        ckpt = tracking_util.Checkpoint(iterator=iterator)
        for _ in range(break_point):
            next(iterator)
        with self.assertRaises(error):
            ckpt.save(self._ckpt_path())
    else:
        with ops.Graph().as_default() as g:
            init_op, get_next_op, saver = self._build_graph(ds_fn, sparse_tensors=sparse_tensors)
            get_next_op = remove_variants(get_next_op)
            with self.session(graph=g) as sess:
                self._initialize(init_op, sess)
                for _ in range(break_point):
                    sess.run(get_next_op)
                with self.assertRaises(error):
                    self._save(sess, saver)