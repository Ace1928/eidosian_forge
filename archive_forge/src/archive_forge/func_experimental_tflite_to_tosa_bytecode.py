from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def experimental_tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant=False, ordered_input_arrays=None, ordered_output_arrays=None):
    if ordered_input_arrays is None:
        ordered_input_arrays = []
    if ordered_output_arrays is None:
        ordered_output_arrays = []
    return ExperimentalTFLiteToTosaBytecode(flatbuffer.encode('utf-8'), bytecode.encode('utf-8'), use_external_constant, ordered_input_arrays, ordered_output_arrays)