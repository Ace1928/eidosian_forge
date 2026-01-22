from typing import Dict, Optional, Tuple, TYPE_CHECKING
from cirq import circuits, ops
def routed_circuit_with_mapping(routed_circuit: 'cirq.AbstractCircuit', initial_map: Optional[Dict['cirq.Qid', 'cirq.Qid']]=None) -> 'cirq.AbstractCircuit':
    """Returns the same circuits with information about the permutation of qubits after each swap.

    Args:
        routed_circuit: a routed circuit that potentially has inserted swaps tagged with a
            RoutingSwapTag.
        initial_map: the initial mapping from logical to physical qubits. If this is not specified
            then the identity mapping of the qubits in routed_circuit will be used as initial_map.

    Raises:
        ValueError: if a non-SWAP gate is tagged with a RoutingSwapTag.
    """
    all_qubits = sorted(routed_circuit.all_qubits())
    qdict = {q: q for q in all_qubits}
    if initial_map is None:
        initial_map = qdict.copy()
    inverse_map = {v: k for k, v in initial_map.items()}

    def swap_print_moment() -> 'cirq.Operation':
        return _SwapPrintGate(tuple(zip(qdict.values(), [inverse_map[x] for x in qdict.values()]))).on(*all_qubits)
    ret_circuit = circuits.Circuit(swap_print_moment())
    for m in routed_circuit:
        swap_in_moment = False
        for op in m:
            if ops.RoutingSwapTag() in op.tags:
                if type(op.gate) != ops.swap_gates.SwapPowGate:
                    raise ValueError('Invalid circuit. A non-SWAP gate cannot be tagged a RoutingSwapTag.')
                swap_in_moment = True
                q1, q2 = op.qubits
                qdict[q1], qdict[q2] = (qdict[q2], qdict[q1])
        ret_circuit.append(m)
        if swap_in_moment:
            ret_circuit.append(swap_print_moment())
    return ret_circuit