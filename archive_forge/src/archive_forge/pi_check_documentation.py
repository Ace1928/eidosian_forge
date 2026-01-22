import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.exceptions import QiskitError
Computes if a number is close to an integer
    fraction or multiple of PI and returns the
    corresponding string.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.
        output (str): Options are 'text' (default),
                      'latex', 'mpl', and 'qasm'.
        ndigits (int or None): Number of digits to print
                               if returning raw inpt.
                               If `None` (default), Python's
                               default float formatting is used.

    Returns:
        str: string representation of output.

    Raises:
        QiskitError: if output is not a valid option.
    