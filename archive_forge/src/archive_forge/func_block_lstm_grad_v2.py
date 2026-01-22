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
def block_lstm_grad_v2(seq_len_max: _atypes.TensorFuzzingAnnotation[_atypes.Int64], x: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], cs_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], w: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], wci: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], wcf: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], wco: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], b: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], i: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], cs: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], f: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], o: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], ci: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], co: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], h: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], cs_grad: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], h_grad: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMGradV2_T], use_peephole: bool, name=None):
    """Computes the LSTM cell backward propagation for the entire time sequence.

  This implementation is to be used in conjunction of BlockLSTMV2.

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`.
      The input gate over the whole time sequence.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh over the whole time sequence.
    f: A `Tensor`. Must have the same type as `x`.
      The forget gate over the whole time sequence.
    o: A `Tensor`. Must have the same type as `x`.
      The output gate over the whole time sequence.
    ci: A `Tensor`. Must have the same type as `x`.
      The cell input over the whole time sequence.
    co: A `Tensor`. Must have the same type as `x`.
      The cell after the tanh over the whole time sequence.
    h: A `Tensor`. Must have the same type as `x`.
      The output h vector over the whole time sequence.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad, b_grad).

    x_grad: A `Tensor`. Has the same type as `x`.
    cs_prev_grad: A `Tensor`. Has the same type as `x`.
    h_prev_grad: A `Tensor`. Has the same type as `x`.
    w_grad: A `Tensor`. Has the same type as `x`.
    wci_grad: A `Tensor`. Has the same type as `x`.
    wcf_grad: A `Tensor`. Has the same type as `x`.
    wco_grad: A `Tensor`. Has the same type as `x`.
    b_grad: A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BlockLSTMGradV2', name, seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad, 'use_peephole', use_peephole)
            _result = _BlockLSTMGradV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return block_lstm_grad_v2_eager_fallback(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, h, cs_grad, h_grad, use_peephole=use_peephole, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    use_peephole = _execute.make_bool(use_peephole, 'use_peephole')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BlockLSTMGradV2', seq_len_max=seq_len_max, x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, i=i, cs=cs, f=f, o=o, ci=ci, co=co, h=h, cs_grad=cs_grad, h_grad=h_grad, use_peephole=use_peephole, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('use_peephole', _op._get_attr_bool('use_peephole'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BlockLSTMGradV2', _inputs_flat, _attrs, _result)
    _result = _BlockLSTMGradV2Output._make(_result)
    return _result