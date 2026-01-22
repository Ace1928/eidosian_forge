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
def random_crop(image: _atypes.TensorFuzzingAnnotation[TV_RandomCrop_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed: int=0, seed2: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RandomCrop_T]:
    """Randomly crop `image`.

  `size` is a 1-D int64 tensor with 2 elements representing the crop height and
  width.  The values must be non negative.

  This Op picks a random location in `image` and crops a `height` by `width`
  rectangle from that location.  The random location is picked so the cropped
  area will fit inside the original image.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      3-D of shape `[height, width, channels]`.
    size: A `Tensor` of type `int64`.
      1-D of length 2 containing: `crop_height`, `crop_width`..
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `image`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RandomCrop', name, image, size, 'seed', seed, 'seed2', seed2)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return random_crop_eager_fallback(image, size, seed=seed, seed2=seed2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RandomCrop', image=image, size=size, seed=seed, seed2=seed2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RandomCrop', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result