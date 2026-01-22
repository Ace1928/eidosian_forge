from typing import Optional, Tuple
from cirq import ops, protocols
def multigate_qcircuit_diagram_info(op: ops.Operation, args: protocols.CircuitDiagramInfoArgs) -> Optional[protocols.CircuitDiagramInfo]:
    if not (isinstance(op, ops.GateOperation) and isinstance(op.gate, ops.InterchangeableQubitsGate)):
        return None
    multigate_parameters = get_multigate_parameters(args)
    if multigate_parameters is None:
        return None
    info = protocols.circuit_diagram_info(op, args, default=None)
    min_index, n_qubits = multigate_parameters
    name = escape_text_for_latex(str(op.gate).rsplit('**', 1)[0] if isinstance(op, ops.GateOperation) else str(op))
    if info is not None and info.exponent != 1:
        name += '^{' + str(info.exponent) + '}'
    box = '\\multigate{' + str(n_qubits - 1) + '}{' + name + '}'
    ghost = '\\ghost{' + name + '}'
    assert args.label_map is not None
    assert args.known_qubits is not None
    symbols = tuple((box if args.label_map[q] == min_index else ghost for q in args.known_qubits))
    return protocols.CircuitDiagramInfo(symbols, connected=False)