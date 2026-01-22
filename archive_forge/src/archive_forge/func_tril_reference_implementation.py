import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def tril_reference_implementation(x, k=0):
    return np.tril(x, k)