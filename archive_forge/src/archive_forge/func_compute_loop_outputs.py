from typing import Any, List
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def compute_loop_outputs(x, seq, trip_count):
    for i in range(trip_count):
        if seq is None:
            seq = []
        seq += [x[:int(i + 1)]]
    return seq