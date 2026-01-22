import contextlib
import pennylane as qml
from pennylane.operation import (
def custom_decomp_expand(self, circuit, max_expansion=decomp_depth):
    with _custom_decomp_context(custom_decomps):
        return custom_fn(circuit, max_expansion=max_expansion)