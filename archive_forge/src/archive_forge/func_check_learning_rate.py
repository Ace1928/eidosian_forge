from copy import copy
import numpy as np
from scipy.stats import multinomial
import pennylane as qml
from .gradient_descent import GradientDescentOptimizer
def check_learning_rate(self, coeffs):
    """Verifies that the learning rate is less than 2 over the Lipschitz constant,
        where the Lipschitz constant is given by :math:`\\sum |c_i|` for Hamiltonian
        coefficients :math:`c_i`.

        Args:
            coeffs (Sequence[float]): the coefficients of the terms in the Hamiltonian

        Raises:
            ValueError: if the learning rate is large than :math:`2/\\sum |c_i|`
        """
    self.lipschitz = np.sum(np.abs(coeffs))
    if self.stepsize > 2 / self.lipschitz:
        raise ValueError(f'The learning rate must be less than {2 / self.lipschitz}')