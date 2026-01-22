from typing import Dict, List, Tuple, TYPE_CHECKING
import cirq
def left_of(qubit: cirq.GridQubit) -> cirq.GridQubit:
    """Gives qubit with one unit less on the first coordinate.

    Args:
        qubit: Reference qubit.

    Returns:
        New translated qubit.
    """
    return cirq.GridQubit(qubit.row - 1, qubit.col)