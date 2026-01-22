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
def model_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], output_types, output_shapes, algorithm: int=0, cpu_budget: int=0, ram_budget: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Identity transformation that models performance.

  Identity transformation that models performance.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    algorithm: An optional `int`. Defaults to `0`.
    cpu_budget: An optional `int`. Defaults to `0`.
    ram_budget: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ModelDataset', name, input_dataset, 'algorithm', algorithm, 'cpu_budget', cpu_budget, 'ram_budget', ram_budget, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return model_dataset_eager_fallback(input_dataset, algorithm=algorithm, cpu_budget=cpu_budget, ram_budget=ram_budget, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'model_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'model_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if algorithm is None:
        algorithm = 0
    algorithm = _execute.make_int(algorithm, 'algorithm')
    if cpu_budget is None:
        cpu_budget = 0
    cpu_budget = _execute.make_int(cpu_budget, 'cpu_budget')
    if ram_budget is None:
        ram_budget = 0
    ram_budget = _execute.make_int(ram_budget, 'ram_budget')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ModelDataset', input_dataset=input_dataset, output_types=output_types, output_shapes=output_shapes, algorithm=algorithm, cpu_budget=cpu_budget, ram_budget=ram_budget, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('algorithm', _op._get_attr_int('algorithm'), 'cpu_budget', _op._get_attr_int('cpu_budget'), 'ram_budget', _op._get_attr_int('ram_budget'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ModelDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result