import cirq
import numpy as np
import pytest
def test_kraus_channel_repr():
    ops = [np.array([[1, 1], [1, 1]], dtype=np.complex64) * 0.5, np.array([[1, -1], [-1, 1]], dtype=np.complex64) * 0.5]
    x_meas = cirq.KrausChannel(ops, key='x_meas')
    assert repr(x_meas) == "cirq.KrausChannel(kraus_ops=[np.array([[(0.5+0j), (0.5+0j)], [(0.5+0j), (0.5+0j)]], dtype=np.dtype('complex64')), np.array([[(0.5+0j), (-0.5+0j)], [(-0.5+0j), (0.5+0j)]], dtype=np.dtype('complex64'))], key='x_meas')"