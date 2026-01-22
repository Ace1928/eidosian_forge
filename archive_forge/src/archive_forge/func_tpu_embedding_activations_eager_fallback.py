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
def tpu_embedding_activations_eager_fallback(embedding_variable: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sliced_activations: _atypes.TensorFuzzingAnnotation[_atypes.Float32], table_id: int, lookup_id: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    table_id = _execute.make_int(table_id, 'table_id')
    lookup_id = _execute.make_int(lookup_id, 'lookup_id')
    embedding_variable = _ops.convert_to_tensor(embedding_variable, _dtypes.float32)
    sliced_activations = _ops.convert_to_tensor(sliced_activations, _dtypes.float32)
    _inputs_flat = [embedding_variable, sliced_activations]
    _attrs = ('table_id', table_id, 'lookup_id', lookup_id)
    _result = _execute.execute(b'TPUEmbeddingActivations', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUEmbeddingActivations', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result