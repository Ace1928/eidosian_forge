import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
input_shape: [1, 1, 4, 4, 4]
        output_shape: [1, 1, 2, 2, 2]
        