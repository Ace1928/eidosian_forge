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
@convert_phase(Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_GRAPHDEF)
def convert_graphdef(input_data, input_tensors, output_tensors, **kwargs):
    """Convert a frozen GraphDef model using the TF Lite converter.

  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
   input_tensors: List of input tensors. Type and shape are computed using
     `foo.shape` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_conversion_flags`.
  """
    model_flags = build_model_flags(**kwargs)
    conversion_flags = build_conversion_flags(**kwargs)
    saved_model_dir = kwargs.get('saved_model_dir', None)
    input_shapes = kwargs.get('input_shapes', None)
    enable_mlir_converter = kwargs.get('enable_mlir_converter', True)
    quantized_input_stats = kwargs.get('quantized_input_stats', None)
    debug_info = kwargs.get('debug_info', None)
    for idx, input_tensor in enumerate(input_tensors):
        input_array = model_flags.input_arrays.add()
        if saved_model_dir:
            input_array.name = input_tensor.name
        else:
            input_array.name = util.get_tensor_name(input_tensor)
        input_array.data_type = convert_tensor_tf_type_to_tflite_type(input_tensor.dtype, usage='input type of the TensorFlow model')
        if _is_quantized_input_stats_required(conversion_flags):
            if quantized_input_stats:
                input_array.mean_value, input_array.std_value = quantized_input_stats[idx]
            else:
                warnings.warn('Statistics for quantized inputs were expected, but not specified; continuing anyway.')
        if input_shapes is None:
            shape = input_tensor.shape
        else:
            shape = input_shapes[idx]
        if shape.rank is not None:
            dims = []
            for dim in shape:
                if dim is None or (isinstance(dim, tensor_shape.Dimension) and dim.value is None):
                    dims.append(-1)
                else:
                    dims.append(int(dim))
            input_array.shape.dims.extend(dims)
            input_array.shape.unknown_rank = False
        else:
            input_array.shape.unknown_rank = True
    for output_tensor in output_tensors:
        if saved_model_dir:
            model_flags.output_arrays.append(output_tensor.name)
        else:
            model_flags.output_arrays.append(util.get_tensor_name(output_tensor))
    data = convert(model_flags, conversion_flags, input_data.SerializeToString(), debug_info_str=debug_info.SerializeToString() if debug_info else None, enable_mlir_converter=enable_mlir_converter)
    return data