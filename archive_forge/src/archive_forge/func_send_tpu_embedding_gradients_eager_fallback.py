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
def send_tpu_embedding_gradients_eager_fallback(inputs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], learning_rates: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], config: str, name, ctx):
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'send_tpu_embedding_gradients' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(learning_rates, (list, tuple)):
        raise TypeError("Expected list for 'learning_rates' argument to 'send_tpu_embedding_gradients' Op, not %r." % learning_rates)
    _attr_NN = len(learning_rates)
    config = _execute.make_str(config, 'config')
    inputs = _ops.convert_n_to_tensor(inputs, _dtypes.float32)
    learning_rates = _ops.convert_n_to_tensor(learning_rates, _dtypes.float32)
    _inputs_flat = list(inputs) + list(learning_rates)
    _attrs = ('N', _attr_N, 'NN', _attr_NN, 'config', config)
    _result = _execute.execute(b'SendTPUEmbeddingGradients', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result