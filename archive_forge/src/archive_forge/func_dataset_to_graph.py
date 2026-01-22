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
def dataset_to_graph(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], stateful_whitelist=[], allow_stateful: bool=False, strip_device_assignment: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Returns a serialized GraphDef representing `input_dataset`.

  Returns a graph representation for `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return the graph representation for.
    stateful_whitelist: An optional list of `strings`. Defaults to `[]`.
    allow_stateful: An optional `bool`. Defaults to `False`.
    strip_device_assignment: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DatasetToGraph', name, input_dataset, 'stateful_whitelist', stateful_whitelist, 'allow_stateful', allow_stateful, 'strip_device_assignment', strip_device_assignment)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dataset_to_graph_eager_fallback(input_dataset, stateful_whitelist=stateful_whitelist, allow_stateful=allow_stateful, strip_device_assignment=strip_device_assignment, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if stateful_whitelist is None:
        stateful_whitelist = []
    if not isinstance(stateful_whitelist, (list, tuple)):
        raise TypeError("Expected list for 'stateful_whitelist' argument to 'dataset_to_graph' Op, not %r." % stateful_whitelist)
    stateful_whitelist = [_execute.make_str(_s, 'stateful_whitelist') for _s in stateful_whitelist]
    if allow_stateful is None:
        allow_stateful = False
    allow_stateful = _execute.make_bool(allow_stateful, 'allow_stateful')
    if strip_device_assignment is None:
        strip_device_assignment = False
    strip_device_assignment = _execute.make_bool(strip_device_assignment, 'strip_device_assignment')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DatasetToGraph', input_dataset=input_dataset, stateful_whitelist=stateful_whitelist, allow_stateful=allow_stateful, strip_device_assignment=strip_device_assignment, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('stateful_whitelist', _op.get_attr('stateful_whitelist'), 'allow_stateful', _op._get_attr_bool('allow_stateful'), 'strip_device_assignment', _op._get_attr_bool('strip_device_assignment'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DatasetToGraph', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result