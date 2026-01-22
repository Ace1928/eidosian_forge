from abc import ABCMeta
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT
@property
def atol(self):
    """Default absolute tolerance parameter for float comparisons."""
    return self.__class__.atol