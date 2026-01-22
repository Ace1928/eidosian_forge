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
def stateless_random_get_key_counter_alg(seed: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGetKeyCounterAlg_Tseed], name=None):
    """Picks the best algorithm based on device, and scrambles seed into key and counter.

  This op picks the best counter-based RNG algorithm based on device, and scrambles a shape-[2] seed into a key and a counter, both needed by the counter-based algorithm. The scrambling is opaque but approximately satisfies the property that different seed results in different key/counter pair (which will in turn result in different random numbers).

  Args:
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, counter, alg).

    key: A `Tensor` of type `uint64`.
    counter: A `Tensor` of type `uint64`.
    alg: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessRandomGetKeyCounterAlg', name, seed)
            _result = _StatelessRandomGetKeyCounterAlgOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_random_get_key_counter_alg_eager_fallback(seed, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessRandomGetKeyCounterAlg', seed=seed, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tseed', _op._get_attr_type('Tseed'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessRandomGetKeyCounterAlg', _inputs_flat, _attrs, _result)
    _result = _StatelessRandomGetKeyCounterAlgOutput._make(_result)
    return _result