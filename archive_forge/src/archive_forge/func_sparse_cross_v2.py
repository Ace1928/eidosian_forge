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
def sparse_cross_v2(indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], values, shapes: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], dense_inputs, sep: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None):
    """Generates sparse cross from a list of sparse and dense tensors.

  The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
  representing features of one feature column. It outputs a 2D `SparseTensor` with
  the batchwise crosses of these features.

  For example, if the inputs are

      inputs[0]: SparseTensor with shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      inputs[1]: SparseTensor with shape = [2, 1]
      [0, 0]: "d"
      [1, 0]: "e"

      inputs[2]: Tensor [["f"], ["g"]]

  then the output will be

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  if hashed_output=true then the output will be

      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))

  Args:
    indices: A list of `Tensor` objects with type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list of `Tensor` objects with types from: `int64`, `string`.
      1-D.   values of each `SparseTensor`.
    shapes: A list with the same length as `indices` of `Tensor` objects with type `int64`.
      1-D.   Shapes of each `SparseTensor`.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      2-D.    Columns represented by dense `Tensor`.
    sep: A `Tensor` of type `string`.
      string used when joining a list of string inputs, can be used as separator later.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor` of type `string`.
    output_shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseCrossV2', name, indices, values, shapes, dense_inputs, sep)
            _result = _SparseCrossV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_cross_v2_eager_fallback(indices, values, shapes, dense_inputs, sep, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(indices, (list, tuple)):
        raise TypeError("Expected list for 'indices' argument to 'sparse_cross_v2' Op, not %r." % indices)
    _attr_N = len(indices)
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'sparse_cross_v2' Op, not %r." % shapes)
    if len(shapes) != _attr_N:
        raise ValueError("List argument 'shapes' to 'sparse_cross_v2' Op with length %d must match length %d of argument 'indices'." % (len(shapes), _attr_N))
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseCrossV2', indices=indices, values=values, shapes=shapes, dense_inputs=dense_inputs, sep=sep, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'sparse_types', _op.get_attr('sparse_types'), 'dense_types', _op.get_attr('dense_types'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseCrossV2', _inputs_flat, _attrs, _result)
    _result = _SparseCrossV2Output._make(_result)
    return _result