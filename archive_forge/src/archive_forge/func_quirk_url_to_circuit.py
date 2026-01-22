import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def quirk_url_to_circuit(quirk_url: str, *, qubits: Optional[Sequence['cirq.Qid']]=None, extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable['cirq.interop.quirk.cells.CellMaker']]=(), max_operation_count: int=10 ** 6) -> 'cirq.Circuit':
    """Parses a Cirq circuit out of a Quirk URL.

    Args:
        quirk_url: The URL of a bookmarked Quirk circuit. It is not required
            that the domain be "algassert.com/quirk". The only important part of
            the URL is the fragment (the part after the #).
        qubits: Qubits to use in the circuit. The length of the list must be
            at least the number of qubits in the Quirk circuit (including unused
            qubits). The maximum number of qubits in a Quirk circuit is 16.
            This argument defaults to `cirq.LineQubit.range(16)` when not
            specified.
        extra_cell_makers: Non-standard Quirk cells to accept. This can be
            used to parse URLs that come from a modified version of Quirk that
            includes gates that Quirk doesn't define. This can be specified
            as either a list of `cirq.interop.quirk.cells.CellMaker` instances,
            or for more simple cases as a dictionary from a Quirk id string
            to a cirq Gate.
        max_operation_count: If the number of operations in the circuit would
            exceed this value, the method raises a `ValueError` instead of
            attempting to construct the circuit. This is important to specify
            for servers parsing unknown input, because Quirk's format allows for
            a billion laughs attack in the form of nested custom gates.

    Examples:

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}'
    ... ))
    0: ───H───@───
              │
    1: ───────X───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
    ...     qubits=[cirq.NamedQubit('Alice'), cirq.NamedQubit('Bob')]
    ... ))
    Alice: ───H───@───
                  │
    Bob: ─────────X───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
    ...     extra_cell_makers={'iswap': cirq.ISWAP}))
    0: ───iSwap───
          │
    1: ───iSwap───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
    ...     extra_cell_makers=[
    ...         cirq.interop.quirk.cells.CellMaker(
    ...             identifier='iswap',
    ...             size=2,
    ...             maker=lambda args: cirq.ISWAP(*args.qubits))
    ...     ]))
    0: ───iSwap───
          │
    1: ───iSwap───

    Returns:
        The parsed circuit.

    Raises:
        ValueError: Invalid circuit URL, or circuit would be larger than
            `max_operations_count`.
    """
    parsed_url = urllib.parse.urlparse(quirk_url)
    if not parsed_url.fragment:
        return circuits.Circuit()
    if not parsed_url.fragment.startswith('circuit='):
        raise ValueError(f'Not a valid quirk url. The URL fragment (the part after the #) must start with "circuit=".\nURL={quirk_url}')
    json_text = parsed_url.fragment[len('circuit='):]
    if '%22' in json_text:
        json_text = urllib.parse.unquote(json_text)
    data = json.loads(json_text)
    return quirk_json_to_circuit(data, qubits=qubits, extra_cell_makers=extra_cell_makers, quirk_url=quirk_url, max_operation_count=max_operation_count)