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
def deduplicate_readonly_buffers(tflite_model):
    """Generates a new model byte array after deduplicating readonly buffers.

  This function should be invoked after the model optimization toolkit. The
  model optimization toolkit assumes that each tensor object owns its each
  buffer separately.

  Args:
    tflite_model: TFLite flatbuffer in a byte array to be deduplicated.

  Returns:
    TFLite flatbuffer in a bytes array, processed with the deduplication method.
  """
    model = flatbuffer_utils.convert_bytearray_to_object(tflite_model)
    read_only_buffer_indices = set()
    for subgraph in model.subgraphs:
        read_only_input_tensor_indices = set()
        for op in subgraph.operators:
            if op.inputs is None:
                continue
            for i, input_tensor_idx in enumerate(op.inputs):
                if op.mutatingVariableInputs is not None:
                    if i < len(op.mutatingVariableInputs) and op.mutatingVariableInputs[i]:
                        continue
                if subgraph.tensors[input_tensor_idx].isVariable:
                    continue
                read_only_input_tensor_indices.add(input_tensor_idx)
        for op in subgraph.operators:
            if op.outputs is not None:
                for output_tensor_idx in op.outputs:
                    read_only_input_tensor_indices.discard(output_tensor_idx)
            if op.intermediates is not None:
                for intermediate_tensor_idx in op.intermediates:
                    read_only_input_tensor_indices.discard(intermediate_tensor_idx)
        if subgraph.inputs is not None:
            for input_tensor_idx in subgraph.inputs:
                read_only_input_tensor_indices.discard(input_tensor_idx)
        if subgraph.outputs is not None:
            for output_tensor_idx in subgraph.outputs:
                read_only_input_tensor_indices.discard(output_tensor_idx)
        for tensor_idx in read_only_input_tensor_indices:
            read_only_buffer_indices.add(subgraph.tensors[tensor_idx].buffer)
    for buffer_idx in read_only_buffer_indices.copy():
        if buffer_idx < 0 or (model.buffers[buffer_idx].data is None or isinstance(model.buffers[buffer_idx].data, list) or model.buffers[buffer_idx].data.size == 0):
            read_only_buffer_indices.discard(buffer_idx)

    class BufferIndex:
        """A class to store index, size, hash of the buffers in TFLite model."""

        def __init__(self, idx, size, hash_value):
            self.idx = idx
            self.size = size
            self.hash_value = hash_value
    read_only_buffers = list(map(lambda index: BufferIndex(index, model.buffers[index].data.size, hashlib.md5(model.buffers[index].data.data.tobytes()).hexdigest()), read_only_buffer_indices))
    read_only_buffers = sorted(read_only_buffers, key=lambda buffer: (buffer.size, buffer.hash_value), reverse=True)
    duplicate_buffer_map = {}
    for i, buffer_i in enumerate(read_only_buffers):
        if buffer_i.idx in duplicate_buffer_map:
            continue
        for buffer_j in read_only_buffers[i + 1:]:
            if buffer_j.idx in duplicate_buffer_map:
                continue
            if buffer_i.size != buffer_j.size:
                break
            if buffer_i.hash_value != buffer_j.hash_value:
                continue
            duplicate_buffer_map[buffer_j.idx] = buffer_i.idx
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if op.inputs is None:
                continue
            for input_tensor in op.inputs:
                buffer_idx = subgraph.tensors[input_tensor].buffer
                if buffer_idx in duplicate_buffer_map:
                    subgraph.tensors[input_tensor].buffer = duplicate_buffer_map[buffer_idx]
    for idx in duplicate_buffer_map:
        model.buffers[idx].data = None
    return flatbuffer_utils.convert_object_to_bytearray(model)