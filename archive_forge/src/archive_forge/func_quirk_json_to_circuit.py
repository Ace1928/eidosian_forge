import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def quirk_json_to_circuit(data: dict, *, qubits: Optional[Sequence['cirq.Qid']]=None, extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable['cirq.interop.quirk.cells.CellMaker']]=(), quirk_url: Optional[str]=None, max_operation_count: int=10 ** 6) -> 'cirq.Circuit':
    """Constructs a Cirq circuit from Quirk's JSON format.

    Args:
        data: Data parsed from quirk's JSON representation.
        qubits: Qubits to use in the circuit. See quirk_url_to_circuit.
        extra_cell_makers: Non-standard Quirk cells to accept. See
            quirk_url_to_circuit.
        quirk_url: If given, the original URL from which the JSON was parsed, as
            described in quirk_url_to_circuit.
        max_operation_count: If the number of operations in the circuit would
            exceed this value, the method raises a `ValueError` instead of
            attempting to construct the circuit. This is important to specify
            for servers parsing unknown input, because Quirk's format allows for
            a billion laughs attack in the form of nested custom gates.

    Examples:

    >>> print(cirq.quirk_json_to_circuit(
    ...     {"cols":[["H"], ["•", "X"]]}
    ... ))
    0: ───H───@───
              │
    1: ───────X───

    Returns:
        The parsed circuit.

    Raises:
        ValueError: Invalid circuit URL, or circuit would be larger than
            `max_operations_count`.
    """

    def msg(error):
        if quirk_url is not None:
            return f'{error}\nURL={quirk_url}\nJSON={data}'
        else:
            return f'{error}\nJSON={data}'
    if not isinstance(data, dict):
        raise ValueError(msg('Circuit JSON must have a top-level dictionary.'))
    if not data.keys() <= {'cols', 'gates', 'init'}:
        raise ValueError(msg('Unrecognized Circuit JSON keys.'))
    if isinstance(extra_cell_makers, Mapping):
        extra_makers = [CellMaker(identifier=identifier, size=protocols.num_qubits(gate), maker=(lambda gate: lambda args: gate(*args.qubits))(gate)) for identifier, gate in extra_cell_makers.items()]
    else:
        extra_makers = list(extra_cell_makers)
    registry = {entry.identifier: entry for entry in [*generate_all_quirk_cell_makers(), *extra_makers]}
    if 'gates' in data:
        if not isinstance(data['gates'], list):
            raise ValueError('"gates" JSON must be a list.')
        for custom_gate in data['gates']:
            _register_custom_gate(custom_gate, registry)
    comp = _parse_cols_into_composite_cell(data, registry)
    if max_operation_count is not None and comp.gate_count() > max_operation_count:
        raise ValueError(f'Quirk URL specifies a circuit with {comp.gate_count()} operations, but max_operation_count={max_operation_count}.')
    circuit = comp.circuit()
    circuit.insert(0, _init_ops(data))
    if qubits is not None:
        qs = qubits

        def map_qubit(qubit: 'cirq.Qid') -> 'cirq.Qid':
            q = cast(devices.LineQubit, qubit)
            if q.x >= len(qs):
                raise IndexError(f'Only {len(qs)} qubits specified, but the given quirk circuit used the qubit at offset {q.x}. Provide more qubits.')
            return qs[q.x]
        circuit = circuit.transform_qubits(map_qubit)
    return circuit