import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
def test_inference_with_attribute(self) -> None:
    model_script = '\n        <\n            ir_version: 8,\n            opset_import: ["" : 18, "custom" : 1],\n            producer_name: "",\n            producer_version: "1.0"\n        >\n        MeanVarianceNormalization (float[N] x) => (float[M] y)\n        {\n            y = custom.custom_mvn <axes = [0]> (x)\n        }\n        <\n            domain: "custom",\n            opset_import: ["" : 18]\n        >\n        custom_mvn <axes>(X) => (Y)\n        {\n          Exponent = Constant <value = float {2.0}>()\n          Epsilon = Constant <value = float {1e-9}>()\n          axes = Constant <value_ints: ints = @axes>()\n          X_RM = ReduceMean (X, axes)\n          EX_squared = Pow (X_RM, Exponent)\n          X_squared = Pow (X, Exponent)\n          E_Xsquared = ReduceMean (X_squared, axes)\n          Variance = Sub (E_Xsquared, EX_squared)\n          STD = Sqrt (Variance)\n          X_variance = Sub (X, X_RM)\n          Processed_STD = Add (STD, Epsilon)\n          Y = Div (X_variance, Processed_STD)\n        }\n        '
    model = onnx.parser.parse_model(model_script)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)