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
def load_tpu_embedding_adagrad_parameters_eager_fallback(parameters: _atypes.TensorFuzzingAnnotation[_atypes.Float32], accumulators: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
    num_shards = _execute.make_int(num_shards, 'num_shards')
    shard_id = _execute.make_int(shard_id, 'shard_id')
    if table_id is None:
        table_id = -1
    table_id = _execute.make_int(table_id, 'table_id')
    if table_name is None:
        table_name = ''
    table_name = _execute.make_str(table_name, 'table_name')
    if config is None:
        config = ''
    config = _execute.make_str(config, 'config')
    parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
    accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
    _inputs_flat = [parameters, accumulators]
    _attrs = ('table_id', table_id, 'table_name', table_name, 'num_shards', num_shards, 'shard_id', shard_id, 'config', config)
    _result = _execute.execute(b'LoadTPUEmbeddingAdagradParameters', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result