from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def compute_choi(channel: cirq.SupportsKraus) -> np.ndarray:
    ks = cirq.kraus(channel)
    d_out, d_in = ks[0].shape
    d = d_in * d_out
    c = np.zeros((d, d), dtype=np.complex128)
    for e in generate_standard_operator_basis(d_in, d_in):
        c += np.kron(apply_channel(channel, e), e)
    return c