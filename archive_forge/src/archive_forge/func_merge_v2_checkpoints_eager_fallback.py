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
def merge_v2_checkpoints_eager_fallback(checkpoint_prefixes: _atypes.TensorFuzzingAnnotation[_atypes.String], destination_prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], delete_old_dirs: bool, allow_missing_files: bool, name, ctx):
    if delete_old_dirs is None:
        delete_old_dirs = True
    delete_old_dirs = _execute.make_bool(delete_old_dirs, 'delete_old_dirs')
    if allow_missing_files is None:
        allow_missing_files = False
    allow_missing_files = _execute.make_bool(allow_missing_files, 'allow_missing_files')
    checkpoint_prefixes = _ops.convert_to_tensor(checkpoint_prefixes, _dtypes.string)
    destination_prefix = _ops.convert_to_tensor(destination_prefix, _dtypes.string)
    _inputs_flat = [checkpoint_prefixes, destination_prefix]
    _attrs = ('delete_old_dirs', delete_old_dirs, 'allow_missing_files', allow_missing_files)
    _result = _execute.execute(b'MergeV2Checkpoints', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result