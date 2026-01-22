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
def tensor_list_concat_v2_eager_fallback(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], element_shape: _atypes.TensorFuzzingAnnotation[TV_TensorListConcatV2_shape_type], leading_dims: _atypes.TensorFuzzingAnnotation[_atypes.Int64], element_dtype: TV_TensorListConcatV2_element_dtype, name, ctx):
    element_dtype = _execute.make_type(element_dtype, 'element_dtype')
    _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64])
    input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
    leading_dims = _ops.convert_to_tensor(leading_dims, _dtypes.int64)
    _inputs_flat = [input_handle, element_shape, leading_dims]
    _attrs = ('element_dtype', element_dtype, 'shape_type', _attr_shape_type)
    _result = _execute.execute(b'TensorListConcatV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorListConcatV2', _inputs_flat, _attrs, _result)
    _result = _TensorListConcatV2Output._make(_result)
    return _result