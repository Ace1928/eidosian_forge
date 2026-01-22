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
def scan_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, preserve_cardinality: bool=False, use_default_device: bool=True, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset successively reduces `f` over the elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    initial_state: A list of `Tensor` objects.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    use_default_device: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ScanDataset', name, input_dataset, initial_state, other_arguments, 'f', f, 'output_types', output_types, 'output_shapes', output_shapes, 'preserve_cardinality', preserve_cardinality, 'use_default_device', use_default_device, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return scan_dataset_eager_fallback(input_dataset, initial_state, other_arguments, f=f, output_types=output_types, output_shapes=output_shapes, preserve_cardinality=preserve_cardinality, use_default_device=use_default_device, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'scan_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'scan_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if preserve_cardinality is None:
        preserve_cardinality = False
    preserve_cardinality = _execute.make_bool(preserve_cardinality, 'preserve_cardinality')
    if use_default_device is None:
        use_default_device = True
    use_default_device = _execute.make_bool(use_default_device, 'use_default_device')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ScanDataset', input_dataset=input_dataset, initial_state=initial_state, other_arguments=other_arguments, f=f, output_types=output_types, output_shapes=output_shapes, preserve_cardinality=preserve_cardinality, use_default_device=use_default_device, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'Tstate', _op.get_attr('Tstate'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'preserve_cardinality', _op._get_attr_bool('preserve_cardinality'), 'use_default_device', _op._get_attr_bool('use_default_device'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ScanDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result