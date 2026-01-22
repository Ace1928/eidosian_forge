from typing import Optional, Tuple
from cirq import ops, protocols
def hardcoded_qcircuit_diagram_info(op: ops.Operation) -> Optional[protocols.CircuitDiagramInfo]:
    if not isinstance(op, ops.GateOperation):
        return None
    symbols = ('\\targ',) if op.gate == ops.X else ('\\control', '\\control') if op.gate == ops.CZ else ('\\control', '\\targ') if op.gate == ops.CNOT else ('\\meter',) if isinstance(op.gate, ops.MeasurementGate) else ()
    return protocols.CircuitDiagramInfo(symbols) if symbols else None