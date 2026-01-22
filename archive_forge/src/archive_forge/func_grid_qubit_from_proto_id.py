import re
from typing import TYPE_CHECKING
import cirq
def grid_qubit_from_proto_id(proto_id: str) -> cirq.GridQubit:
    """Parse a proto id to a `cirq.GridQubit`.

    Proto ids for grid qubits are of the form `{row}_{col}` where `{row}` is
    the integer row of the grid qubit, and `{col}` is the integer column of
    the qubit.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.GridQubit` corresponding to the proto id.

    Raises:
        ValueError: If the string not of the correct format.
    """
    match = re.match(GRID_QUBIT_ID_PATTERN, proto_id)
    if match is None:
        raise ValueError(f'GridQubit proto id must be of the form [q]<int>_<int> but was {proto_id}')
    row, col = match.groups()
    return cirq.GridQubit(row=int(row), col=int(col))