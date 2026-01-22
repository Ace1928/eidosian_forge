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
def retrieve_tpu_embedding_ftrl_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
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
    _inputs_flat = []
    _attrs = ('table_id', table_id, 'table_name', table_name, 'num_shards', num_shards, 'shard_id', shard_id, 'config', config)
    _result = _execute.execute(b'RetrieveTPUEmbeddingFTRLParameters', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RetrieveTPUEmbeddingFTRLParameters', _inputs_flat, _attrs, _result)
    _result = _RetrieveTPUEmbeddingFTRLParametersOutput._make(_result)
    return _result