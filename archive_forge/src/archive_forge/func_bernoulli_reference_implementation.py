import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def bernoulli_reference_implementation(x, dtype):
    return np.random.binomial(1, p=x).astype(dtype)