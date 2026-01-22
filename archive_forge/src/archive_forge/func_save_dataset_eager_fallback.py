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
def save_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], shard_func_other_args, shard_func, compression: str, use_shard_func: bool, name, ctx):
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if use_shard_func is None:
        use_shard_func = True
    use_shard_func = _execute.make_bool(use_shard_func, 'use_shard_func')
    _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [input_dataset, path] + list(shard_func_other_args)
    _attrs = ('compression', compression, 'shard_func', shard_func, 'use_shard_func', use_shard_func, 'Tshard_func_args', _attr_Tshard_func_args)
    _result = _execute.execute(b'SaveDataset', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result