import pennylane as qml
from pennylane.operation import Operator
Check if the operation is unitary.

    A matrix is unitary if its adjoint is also its inverse, that is, if

    .. math:: O^\dagger O = OO^\dagger = I

    Args:
        op (~.operation.Operator): the operator to check against

    Returns:
        bool: True if the operation is unitary, False otherwise

    .. note::
        This check might be expensive for large operators.

    **Example**

    >>> op = qml.RX(0.54, wires=0)
    >>> qml.is_unitary(op)
    True
    >>> op2 = op + op
    >>> qml.is_unitary(op2)
    False
    