import pennylane as qml
from pennylane.operation import Operation
from pennylane.math import requires_grad, unwrap
from pennylane.ops import Sum, SProd, Hamiltonian
A method for determining the upper-bound for the error in the approximation of
        the true matrix exponential.

        The error is bounded according to the following expression:

        .. math::

            \epsilon \ \leq \ \frac{2\lambda^{2}t^{2}}{n}  e^{\frac{2 \lambda t}{n}},

        where :math:`t` is time, :math:`\lambda = \sum_j |h_j|` and :math:`n` is the total number of
        terms to be added to the product. For more details see `Phys. Rev. Lett. 123, 070503 (2019) <https://arxiv.org/abs/1811.08017>`_.

        Args:
            hamiltonian (Union[.Hamiltonian, .Sum]): The Hamiltonian written as a sum of operations
            time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{-iHt}`
            n (int): An integer representing the number of exponentiated terms. default is 1

        Raises:
            TypeError: The given operator must be a PennyLane .Hamiltonian or .Sum

        Returns:
            float: upper bound on the precision achievable using the QDrift protocol
        