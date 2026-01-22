import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export
def convert_tensor_tf_type_to_tflite_type(tf_type: dtypes.DType, usage: str='') -> _types_pb2.IODataType:
    """Convert tensor type from tf type to tflite type.

  Args:
    tf_type: TensorFlow type.
    usage: Text describing the reason for invoking this function.

  Raises:
    ValueError: If `tf_type` is unsupported.

  Returns:
    tflite_type: TFLite type. Refer to lite/toco/types.proto.
  """
    mapping = {dtypes.float16: _types_pb2.FLOAT16, dtypes.float32: _types_pb2.FLOAT, dtypes.float64: _types_pb2.FLOAT64, dtypes.int8: _types_pb2.INT8, dtypes.int16: _types_pb2.INT16, dtypes.uint16: _types_pb2.UINT16, dtypes.int32: _types_pb2.INT32, dtypes.int64: _types_pb2.INT64, dtypes.uint8: _types_pb2.UINT8, dtypes.uint32: _types_pb2.UINT32, dtypes.uint64: _types_pb2.UINT64, dtypes.string: _types_pb2.STRING, dtypes.bool: _types_pb2.BOOL, dtypes.complex64: _types_pb2.COMPLEX64, dtypes.complex128: _types_pb2.COMPLEX128}
    tflite_type = mapping.get(tf_type)
    if tflite_type is None:
        raise ValueError('Unsupported TensorFlow type `{0}` provided for the {1}'.format(tf_type, usage))
    return tflite_type