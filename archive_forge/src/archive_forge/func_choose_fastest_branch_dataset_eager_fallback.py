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
def choose_fastest_branch_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], ratio_numerator: _atypes.TensorFuzzingAnnotation[_atypes.Int64], ratio_denominator: _atypes.TensorFuzzingAnnotation[_atypes.Int64], other_arguments, num_elements_per_branch: int, branches, other_arguments_lengths, output_types, output_shapes, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    num_elements_per_branch = _execute.make_int(num_elements_per_branch, 'num_elements_per_branch')
    if not isinstance(branches, (list, tuple)):
        raise TypeError("Expected list for 'branches' argument to 'choose_fastest_branch_dataset' Op, not %r." % branches)
    if not isinstance(other_arguments_lengths, (list, tuple)):
        raise TypeError("Expected list for 'other_arguments_lengths' argument to 'choose_fastest_branch_dataset' Op, not %r." % other_arguments_lengths)
    other_arguments_lengths = [_execute.make_int(_i, 'other_arguments_lengths') for _i in other_arguments_lengths]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'choose_fastest_branch_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'choose_fastest_branch_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    ratio_numerator = _ops.convert_to_tensor(ratio_numerator, _dtypes.int64)
    ratio_denominator = _ops.convert_to_tensor(ratio_denominator, _dtypes.int64)
    _inputs_flat = [input_dataset, ratio_numerator, ratio_denominator] + list(other_arguments)
    _attrs = ('Targuments', _attr_Targuments, 'num_elements_per_branch', num_elements_per_branch, 'branches', branches, 'other_arguments_lengths', other_arguments_lengths, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'ChooseFastestBranchDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ChooseFastestBranchDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result