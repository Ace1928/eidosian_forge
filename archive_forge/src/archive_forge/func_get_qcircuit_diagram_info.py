from typing import Optional, Tuple
from cirq import ops, protocols
def get_qcircuit_diagram_info(op: ops.Operation, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
    info = hardcoded_qcircuit_diagram_info(op)
    if info is None:
        info = multigate_qcircuit_diagram_info(op, args)
    if info is None:
        info = fallback_qcircuit_diagram_info(op, args)
    return info