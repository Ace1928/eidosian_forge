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
def lookup_table_find_v2(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], keys: _atypes.TensorFuzzingAnnotation[TV_LookupTableFindV2_Tin], default_value: _atypes.TensorFuzzingAnnotation[TV_LookupTableFindV2_Tout], name=None) -> _atypes.TensorFuzzingAnnotation[TV_LookupTableFindV2_Tout]:
    """Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'LookupTableFindV2', name, table_handle, keys, default_value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return lookup_table_find_v2_eager_fallback(table_handle, keys, default_value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('LookupTableFindV2', table_handle=table_handle, keys=keys, default_value=default_value, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op._get_attr_type('Tin'), 'Tout', _op._get_attr_type('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LookupTableFindV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result