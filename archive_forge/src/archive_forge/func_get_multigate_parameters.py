from typing import Optional, Tuple
from cirq import ops, protocols
def get_multigate_parameters(args: protocols.CircuitDiagramInfoArgs) -> Optional[Tuple[int, int]]:
    if args.label_map is None or args.known_qubits is None:
        return None
    indices = [args.label_map[q] for q in args.known_qubits]
    min_index = min(indices)
    n_qubits = len(args.known_qubits)
    if sorted(indices) != list(range(min_index, min_index + n_qubits)):
        return None
    return (min_index, n_qubits)