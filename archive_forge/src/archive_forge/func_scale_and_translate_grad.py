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
def scale_and_translate_grad(grads: _atypes.TensorFuzzingAnnotation[TV_ScaleAndTranslateGrad_T], original_image: _atypes.TensorFuzzingAnnotation[TV_ScaleAndTranslateGrad_T], scale: _atypes.TensorFuzzingAnnotation[_atypes.Float32], translation: _atypes.TensorFuzzingAnnotation[_atypes.Float32], kernel_type: str='lanczos3', antialias: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ScaleAndTranslateGrad_T]:
    """TODO: add doc.

  Args:
    grads: A `Tensor`. Must be one of the following types: `float32`.
    original_image: A `Tensor`. Must have the same type as `grads`.
    scale: A `Tensor` of type `float32`.
    translation: A `Tensor` of type `float32`.
    kernel_type: An optional `string`. Defaults to `"lanczos3"`.
    antialias: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grads`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ScaleAndTranslateGrad', name, grads, original_image, scale, translation, 'kernel_type', kernel_type, 'antialias', antialias)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return scale_and_translate_grad_eager_fallback(grads, original_image, scale, translation, kernel_type=kernel_type, antialias=antialias, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if kernel_type is None:
        kernel_type = 'lanczos3'
    kernel_type = _execute.make_str(kernel_type, 'kernel_type')
    if antialias is None:
        antialias = True
    antialias = _execute.make_bool(antialias, 'antialias')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ScaleAndTranslateGrad', grads=grads, original_image=original_image, scale=scale, translation=translation, kernel_type=kernel_type, antialias=antialias, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'kernel_type', _op.get_attr('kernel_type'), 'antialias', _op._get_attr_bool('antialias'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ScaleAndTranslateGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result