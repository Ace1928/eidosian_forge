import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def compute_if_outputs(x, cond):
    if cond:
        return []
    else:
        return x