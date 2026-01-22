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
def tensor_array_split_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], value: _atypes.TensorFuzzingAnnotation[TV_TensorArraySplit_T], lengths: _atypes.TensorFuzzingAnnotation[_atypes.Int64], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    raise RuntimeError("tensor_array_split op does not support eager execution. Arg 'handle' is a ref.")