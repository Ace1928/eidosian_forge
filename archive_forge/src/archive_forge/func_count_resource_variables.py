import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def count_resource_variables(model):
    """Calculates the number of unique resource variables in a model.

  Args:
    model: the input tflite model, either as bytearray or object.

  Returns:
    An integer number representing the number of unique resource variables.
  """
    if not isinstance(model, schema_fb.ModelT):
        model = convert_bytearray_to_object(model)
    unique_shared_names = set()
    for subgraph in model.subgraphs:
        if subgraph.operators is None:
            continue
        for op in subgraph.operators:
            builtin_code = schema_util.get_builtin_code_from_operator_code(model.operatorCodes[op.opcodeIndex])
            if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
                unique_shared_names.add(op.builtinOptions.sharedName)
    return len(unique_shared_names)