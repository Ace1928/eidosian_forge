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
def window_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shift: _atypes.TensorFuzzingAnnotation[_atypes.Int64], stride: _atypes.TensorFuzzingAnnotation[_atypes.Int64], drop_remainder: _atypes.TensorFuzzingAnnotation[_atypes.Bool], output_types, output_shapes, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """  Combines (nests of) input elements into a dataset of (nests of) windows.

  A "window" is a finite dataset of flat elements of size `size` (or possibly
  fewer if there are not enough input elements to fill the window and
  `drop_remainder` evaluates to false).

  The `shift` argument determines the number of input elements by which
  the window moves on each iteration.  The first element in the `k`th window
  will be element

  ```
  1 + (k-1) * shift
  ```

  of the input dataset. In particular, the first element of the first window
  will always be the first element of the input dataset.  

  If the `stride` parameter is greater than 1, then each window will skip
  `(stride - 1)` input elements between each element that appears in the
  window. Output windows will still contain `size` elements regardless of
  the value of `stride`.

  The `stride` argument determines the stride of the input elements, and the
  `shift` argument determines the shift of the window.

  For example, letting `{...}` to represent a Dataset:

  - `tf.data.Dataset.range(7).window(2)` produces
    `{{0, 1}, {2, 3}, {4, 5}, {6}}`
  - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
    `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
  - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
    `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

  Note that when the `window` transformation is applied to a dataset of
  nested elements, it produces a dataset of nested windows.

  For example:

  - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
    produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
  - `tf.data.Dataset.from_tensor_slices({"a": range(4)}).window(2)`
    produces `{{"a": {0, 1}}, {"a": {2, 3}}}`

  Args:
    input_dataset: A `Tensor` of type `variant`.
    size: A `Tensor` of type `int64`.
      An integer scalar, representing the number of elements
      of the input dataset to combine into a window. Must be positive.
    shift: A `Tensor` of type `int64`.
      An integer scalar, representing the number of input elements
      by which the window moves in each iteration.  Defaults to `size`.
      Must be positive.
    stride: A `Tensor` of type `int64`.
      An integer scalar, representing the stride of the input elements
      in the sliding window. Must be positive. The default value of 1 means
      "retain every input element".
    drop_remainder: A `Tensor` of type `bool`.
      A Boolean scalar, representing whether the last window should be
      dropped if its size is smaller than `window_size`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'WindowDataset', name, input_dataset, size, shift, stride, drop_remainder, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return window_dataset_eager_fallback(input_dataset, size, shift, stride, drop_remainder, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'window_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'window_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('WindowDataset', input_dataset=input_dataset, size=size, shift=shift, stride=stride, drop_remainder=drop_remainder, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('WindowDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result