import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.usefixtures('closefigures')
def test_scatter_plot_normalized_kak_interaction_coefficients():
    a, b = cirq.LineQubit.range(2)
    data = [cirq.kak_decomposition(cirq.unitary(cirq.CZ)), cirq.unitary(cirq.CZ), cirq.CZ, cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))]
    ax = cirq.scatter_plot_normalized_kak_interaction_coefficients(data)
    assert ax is not None
    ax2 = cirq.scatter_plot_normalized_kak_interaction_coefficients(data, s=1, c='blue', ax=ax, include_frame=False, label='test')
    assert ax2 is ax
    ax3 = cirq.scatter_plot_normalized_kak_interaction_coefficients(data[1], ax=ax)
    assert ax3 is ax