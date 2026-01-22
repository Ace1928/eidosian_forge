import pytest
import numpy as np
import cirq
def test_block_diag_dtype():
    assert cirq.block_diag().dtype == np.complex128
    assert cirq.block_diag(np.array([[1]], dtype=np.int8)).dtype == np.int8
    assert cirq.block_diag(np.array([[1]], dtype=np.float32), np.array([[2]], dtype=np.float32)).dtype == np.float32
    assert cirq.block_diag(np.array([[1]], dtype=np.float64), np.array([[2]], dtype=np.float64)).dtype == np.float64
    assert cirq.block_diag(np.array([[1]], dtype=np.float32), np.array([[2]], dtype=np.float64)).dtype == np.float64
    assert cirq.block_diag(np.array([[1]], dtype=np.float32), np.array([[2]], dtype=np.complex64)).dtype == np.complex64
    assert cirq.block_diag(np.array([[1]], dtype=int), np.array([[2]], dtype=np.complex128)).dtype == np.complex128