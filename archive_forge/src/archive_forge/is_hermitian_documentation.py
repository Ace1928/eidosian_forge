import pennylane as qml
from pennylane.operation import Operator
Check if the operation is hermitian.

    A hermitian matrix is a complex square matrix that is equal to its own adjoint

    .. math:: O^\dagger = O

    Args:
        op (~.operation.Operator): the operator to check against

    Returns:
        bool: True if the operation is hermitian, False otherwise

    .. note::
        This check might be expensive for large operators.

    **Example**

    >>> op = qml.X(0)
    >>> qml.is_hermitian(op)
    True
    >>> op2 = qml.RX(0.54, wires=0)
    >>> qml.is_hermitian(op2)
    False
    