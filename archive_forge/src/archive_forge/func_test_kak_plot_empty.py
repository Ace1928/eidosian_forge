import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.usefixtures('closefigures')
def test_kak_plot_empty():
    cirq.scatter_plot_normalized_kak_interaction_coefficients([])