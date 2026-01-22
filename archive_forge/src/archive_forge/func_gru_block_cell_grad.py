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
def gru_block_cell_grad(x: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], w_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], w_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], b_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], b_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], r: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], u: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], d_h: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], name=None):
    """Computes the GRU cell back-propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      d_h: Gradients of the h_new wrt to objective function.

  Returns
      d_x: Gradients of the x wrt to objective function.
      d_h_prev: Gradients of the h wrt to objective function.
      d_c_bar Gradients of the c_bar wrt to objective function.
      d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.

  This kernel op implements the following mathematical equations:

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \\circ
  Matrix multiplication is represented by *

  Additional notes for clarity:

  `w_ru` can be segmented into 4 different matrices.
  ```
  w_ru = [w_r_x w_u_x
          w_r_h_prev w_u_h_prev]
  ```
  Similarly, `w_c` can be segmented into 2 different matrices.
  ```
  w_c = [w_c_x w_c_h_prevr]
  ```
  Same goes for biases.
  ```
  b_ru = [b_ru_x b_ru_h]
  b_c = [b_c_x b_c_h]
  ```
  Another note on notation:
  ```
  d_x = d_x_component_1 + d_x_component_2

  where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
  and d_x_component_2 = d_c_bar * w_c_x^T

  d_h_prev = d_h_prev_component_1 + d_h_prevr \\circ r + d_h \\circ u
  where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
  ```

  Mathematics behind the Gradients below:
  ```
  d_c_bar = d_h \\circ (1-u) \\circ (1-c \\circ c)
  d_u_bar = d_h \\circ (h-c) \\circ u \\circ (1-u)

  d_r_bar_u_bar = [d_r_bar d_u_bar]

  [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

  [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

  d_x = d_x_component_1 + d_x_component_2

  d_h_prev = d_h_prev_component_1 + d_h_prevr \\circ r + u
  ```
  Below calculation is performed in the python wrapper for the Gradients
  (not in the gradient kernel.)
  ```
  d_w_ru = x_h_prevr^T * d_c_bar

  d_w_c = x_h_prev^T * d_r_bar_u_bar

  d_b_ru = sum of d_r_bar_u_bar along axis = 0

  d_b_c = sum of d_c_bar along axis = 0
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    r: A `Tensor`. Must have the same type as `x`.
    u: A `Tensor`. Must have the same type as `x`.
    c: A `Tensor`. Must have the same type as `x`.
    d_h: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_x, d_h_prev, d_c_bar, d_r_bar_u_bar).

    d_x: A `Tensor`. Has the same type as `x`.
    d_h_prev: A `Tensor`. Has the same type as `x`.
    d_c_bar: A `Tensor`. Has the same type as `x`.
    d_r_bar_u_bar: A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'GRUBlockCellGrad', name, x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h)
            _result = _GRUBlockCellGradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return gru_block_cell_grad_eager_fallback(x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('GRUBlockCellGrad', x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c, r=r, u=u, c=c, d_h=d_h, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('GRUBlockCellGrad', _inputs_flat, _attrs, _result)
    _result = _GRUBlockCellGradOutput._make(_result)
    return _result