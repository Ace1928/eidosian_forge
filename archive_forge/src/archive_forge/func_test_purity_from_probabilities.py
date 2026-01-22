import numpy as np
from cirq.experiments import purity_from_probabilities
def test_purity_from_probabilities():
    probabilities = np.random.uniform(0, 1, size=4)
    probabilities /= np.sum(probabilities)
    purity = purity_from_probabilities(4, probabilities)
    np.testing.assert_allclose(purity, np.var(probabilities) * 80 / 3)