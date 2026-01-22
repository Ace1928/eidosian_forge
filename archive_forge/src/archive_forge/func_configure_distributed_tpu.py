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
def configure_distributed_tpu(embedding_config: str='', tpu_embedding_config: str='', is_global_init: bool=False, enable_whole_mesh_compilations: bool=False, compilation_failure_closes_chips: bool=True, tpu_cancellation_closes_chips: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Sets up the centralized structures for a distributed TPU system.

  Args:
    embedding_config: An optional `string`. Defaults to `""`.
      Reserved. Do not use.
    tpu_embedding_config: An optional `string`. Defaults to `""`.
      Serialized tensorflow.tpu.TPUEmbeddingConfiguration that
      describes the embedding lookups of the program.
    is_global_init: An optional `bool`. Defaults to `False`.
      Reserved. Do not use.
    enable_whole_mesh_compilations: An optional `bool`. Defaults to `False`.
    compilation_failure_closes_chips: An optional `bool`. Defaults to `True`.
    tpu_cancellation_closes_chips: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ConfigureDistributedTPU', name, 'embedding_config', embedding_config, 'tpu_embedding_config', tpu_embedding_config, 'is_global_init', is_global_init, 'enable_whole_mesh_compilations', enable_whole_mesh_compilations, 'compilation_failure_closes_chips', compilation_failure_closes_chips, 'tpu_cancellation_closes_chips', tpu_cancellation_closes_chips)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return configure_distributed_tpu_eager_fallback(embedding_config=embedding_config, tpu_embedding_config=tpu_embedding_config, is_global_init=is_global_init, enable_whole_mesh_compilations=enable_whole_mesh_compilations, compilation_failure_closes_chips=compilation_failure_closes_chips, tpu_cancellation_closes_chips=tpu_cancellation_closes_chips, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if embedding_config is None:
        embedding_config = ''
    embedding_config = _execute.make_str(embedding_config, 'embedding_config')
    if tpu_embedding_config is None:
        tpu_embedding_config = ''
    tpu_embedding_config = _execute.make_str(tpu_embedding_config, 'tpu_embedding_config')
    if is_global_init is None:
        is_global_init = False
    is_global_init = _execute.make_bool(is_global_init, 'is_global_init')
    if enable_whole_mesh_compilations is None:
        enable_whole_mesh_compilations = False
    enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, 'enable_whole_mesh_compilations')
    if compilation_failure_closes_chips is None:
        compilation_failure_closes_chips = True
    compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, 'compilation_failure_closes_chips')
    if tpu_cancellation_closes_chips is None:
        tpu_cancellation_closes_chips = 0
    tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, 'tpu_cancellation_closes_chips')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ConfigureDistributedTPU', embedding_config=embedding_config, tpu_embedding_config=tpu_embedding_config, is_global_init=is_global_init, enable_whole_mesh_compilations=enable_whole_mesh_compilations, compilation_failure_closes_chips=compilation_failure_closes_chips, tpu_cancellation_closes_chips=tpu_cancellation_closes_chips, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('embedding_config', _op.get_attr('embedding_config'), 'tpu_embedding_config', _op.get_attr('tpu_embedding_config'), 'is_global_init', _op._get_attr_bool('is_global_init'), 'enable_whole_mesh_compilations', _op._get_attr_bool('enable_whole_mesh_compilations'), 'compilation_failure_closes_chips', _op._get_attr_bool('compilation_failure_closes_chips'), 'tpu_cancellation_closes_chips', _op._get_attr_int('tpu_cancellation_closes_chips'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ConfigureDistributedTPU', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result