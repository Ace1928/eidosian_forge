from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def generate_standard_operator_basis(d_out: int, d_in: int) -> Iterable[np.ndarray]:
    for i in range(d_out):
        for j in range(d_in):
            e_ij = np.zeros((d_out, d_in))
            e_ij[i, j] = 1
            yield e_ij