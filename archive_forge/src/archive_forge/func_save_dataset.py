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
def save_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], shard_func_other_args, shard_func, compression: str='', use_shard_func: bool=True, name=None):
    """TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    path: A `Tensor` of type `string`.
    shard_func_other_args: A list of `Tensor` objects.
    shard_func: A function decorated with @Defun.
    compression: An optional `string`. Defaults to `""`.
    use_shard_func: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SaveDataset', name, input_dataset, path, shard_func_other_args, 'compression', compression, 'shard_func', shard_func, 'use_shard_func', use_shard_func)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return save_dataset_eager_fallback(input_dataset, path, shard_func_other_args, compression=compression, shard_func=shard_func, use_shard_func=use_shard_func, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if use_shard_func is None:
        use_shard_func = True
    use_shard_func = _execute.make_bool(use_shard_func, 'use_shard_func')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SaveDataset', input_dataset=input_dataset, path=path, shard_func_other_args=shard_func_other_args, shard_func=shard_func, compression=compression, use_shard_func=use_shard_func, name=name)
    return _op