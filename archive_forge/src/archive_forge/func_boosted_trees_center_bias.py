import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def boosted_trees_center_bias(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mean_gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], mean_hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    """Calculates the prior from the training data (the bias) and fills in the first node with the logits' prior. Returns a boolean indicating whether to continue centering.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble.
    mean_gradients: A `Tensor` of type `float32`.
      A tensor with shape=[logits_dimension] with mean of gradients for a first node.
    mean_hessians: A `Tensor` of type `float32`.
      A tensor with shape=[logits_dimension] mean of hessians for a first node.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesCenterBias', name, tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_center_bias_eager_fallback(tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesCenterBias', tree_ensemble_handle=tree_ensemble_handle, mean_gradients=mean_gradients, mean_hessians=mean_hessians, l1=l1, l2=l2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesCenterBias', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result