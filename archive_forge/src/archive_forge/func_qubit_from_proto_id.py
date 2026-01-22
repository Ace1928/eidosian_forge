import re
from typing import TYPE_CHECKING
import cirq
def qubit_from_proto_id(proto_id: str) -> cirq.Qid:
    """Return a `cirq.Qid` for a proto id.

    Proto IDs of the form {int}_{int} are parsed as GridQubits.

    Proto IDs of the form {int} are parsed as LineQubits.

    All other proto IDs are parsed as NamedQubits. Note that this will happily
    accept any string; for circuits which explicitly use Grid or LineQubits,
    prefer one of the specialized methods below.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.Qid` corresponding to the proto id.
    """
    num_coords = len(proto_id.split('_'))
    if num_coords == 2:
        try:
            grid_q = grid_qubit_from_proto_id(proto_id)
            return grid_q
        except ValueError:
            pass
    elif num_coords == 1:
        try:
            line_q = line_qubit_from_proto_id(proto_id)
            return line_q
        except ValueError:
            pass
    named_q = named_qubit_from_proto_id(proto_id)
    return named_q