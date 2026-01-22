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
def batch_self_adjoint_eig_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_BatchSelfAdjointEigV2_T], compute_v: bool, name, ctx):
    if compute_v is None:
        compute_v = True
    compute_v = _execute.make_bool(compute_v, 'compute_v')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float64, _dtypes.float32])
    _inputs_flat = [input]
    _attrs = ('compute_v', compute_v, 'T', _attr_T)
    _result = _execute.execute(b'BatchSelfAdjointEigV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BatchSelfAdjointEigV2', _inputs_flat, _attrs, _result)
    _result = _BatchSelfAdjointEigV2Output._make(_result)
    return _result