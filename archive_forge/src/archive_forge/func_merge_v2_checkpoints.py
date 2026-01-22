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
def merge_v2_checkpoints(checkpoint_prefixes: _atypes.TensorFuzzingAnnotation[_atypes.String], destination_prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], delete_old_dirs: bool=True, allow_missing_files: bool=False, name=None):
    """V2 format specific: merges the metadata files of sharded checkpoints.  The

  result is one logical checkpoint, with one physical metadata file and renamed
  data files.

  Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.

  If delete_old_dirs is true, attempts to delete recursively the dirname of each
  path in the input checkpoint_prefixes.  This is useful when those paths are non
  user-facing temporary locations.

  If allow_missing_files is true, merges the checkpoint prefixes as long as
  at least one file exists. Otherwise, if no files exist, an error will be thrown.
  The default value for allow_missing_files is false.

  Args:
    checkpoint_prefixes: A `Tensor` of type `string`.
      prefixes of V2 checkpoints to merge.
    destination_prefix: A `Tensor` of type `string`.
      scalar.  The desired final prefix.  Allowed to be the same
      as one of the checkpoint_prefixes.
    delete_old_dirs: An optional `bool`. Defaults to `True`. see above.
    allow_missing_files: An optional `bool`. Defaults to `False`. see above.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MergeV2Checkpoints', name, checkpoint_prefixes, destination_prefix, 'delete_old_dirs', delete_old_dirs, 'allow_missing_files', allow_missing_files)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return merge_v2_checkpoints_eager_fallback(checkpoint_prefixes, destination_prefix, delete_old_dirs=delete_old_dirs, allow_missing_files=allow_missing_files, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if delete_old_dirs is None:
        delete_old_dirs = True
    delete_old_dirs = _execute.make_bool(delete_old_dirs, 'delete_old_dirs')
    if allow_missing_files is None:
        allow_missing_files = False
    allow_missing_files = _execute.make_bool(allow_missing_files, 'allow_missing_files')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MergeV2Checkpoints', checkpoint_prefixes=checkpoint_prefixes, destination_prefix=destination_prefix, delete_old_dirs=delete_old_dirs, allow_missing_files=allow_missing_files, name=name)
    return _op