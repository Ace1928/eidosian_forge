from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def compute_superoperator(channel: cirq.SupportsKraus) -> np.ndarray:
    ks = cirq.kraus(channel)
    d_out, d_in = ks[0].shape
    m = np.zeros((d_out * d_out, d_in * d_in), dtype=np.complex128)
    for k, e_in in enumerate(generate_standard_operator_basis(d_in, d_in)):
        m[:, k] = np.reshape(apply_channel(channel, e_in), d_out * d_out)
    return m