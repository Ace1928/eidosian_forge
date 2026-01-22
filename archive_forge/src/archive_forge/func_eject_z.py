from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from cirq import ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
@transformer_api.transformer(add_deep_support=True)
def eject_z(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, atol: float=0.0, eject_parameterized: bool=False) -> 'cirq.Circuit':
    """Pushes Z gates towards the end of the circuit.

    As the Z gates get pushed they may absorb other Z gates, get absorbed into
    measurements, cross CZ gates, cross PhasedXPowGate (aka W) gates (by phasing them), etc.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          atol: Maximum absolute error tolerance. The optimization is
               permitted to simply drop negligible combinations of Z gates,
               with a threshold determined by this tolerance.
          eject_parameterized: If True, the optimization will attempt to eject
              parameterized Z gates as well.  This may result in other gates
              parameterized by symbolic expressions.
    Returns:
        Copy of the transformed input circuit.
    """
    qubit_phase: Dict[ops.Qid, float] = defaultdict(lambda: 0)
    tags_to_ignore = set(context.tags_to_ignore) if context else set()
    phased_xz_replacements: Dict[Tuple[int, ops.Operation], ops.PhasedXZGate] = {}
    last_phased_xz_op: Dict[ops.Qid, Optional[Tuple[int, ops.Operation]]] = defaultdict(lambda: None)

    def dump_tracked_phase(qubits: Iterable[ops.Qid]) -> 'cirq.OP_TREE':
        """Zeroes qubit_phase entries by emitting Z gates."""
        for q in qubits:
            p, key = (qubit_phase[q], last_phased_xz_op[q])
            qubit_phase[q] = 0
            if not (key or single_qubit_decompositions.is_negligible_turn(p, atol)):
                yield (ops.Z(q) ** (p * 2))
            elif key:
                phased_xz_replacements[key] = phased_xz_replacements[key].with_z_exponent(p * 2)

    def map_func(op: 'cirq.Operation', moment_index: int) -> 'cirq.OP_TREE':
        last_phased_xz_op.update({q: None for q in op.qubits})
        if tags_to_ignore & set(op.tags):
            return [dump_tracked_phase(op.qubits), op]
        gate = op.gate
        if gate is None:
            return [dump_tracked_phase(op.qubits), op]
        if _is_swaplike(gate):
            a, b = op.qubits
            qubit_phase[a], qubit_phase[b] = (qubit_phase[b], qubit_phase[a])
            return op
        if isinstance(gate, ops.MeasurementGate):
            for q in op.qubits:
                qubit_phase[q] = 0
            return op
        if isinstance(gate, ops.ZPowGate) and (eject_parameterized or not protocols.is_parameterized(gate)):
            qubit_phase[op.qubits[0]] += gate.exponent / 2
            return []
        phased_op = op
        for i, p in enumerate([qubit_phase[q] for q in op.qubits]):
            if not single_qubit_decompositions.is_negligible_turn(p, atol):
                phased_op = protocols.phase_by(phased_op, -p, i, default=None)
        if phased_op is None:
            return [dump_tracked_phase(op.qubits), op]
        gate = phased_op.gate
        if isinstance(gate, ops.PhasedXZGate) and (eject_parameterized or not protocols.is_parameterized(gate.z_exponent)):
            qubit = phased_op.qubits[0]
            qubit_phase[qubit] += gate.z_exponent / 2
            gate = gate.with_z_exponent(0)
            phased_op = gate.on(qubit)
            phased_xz_replacements[moment_index, phased_op] = gate
            last_phased_xz_op[qubit] = (moment_index, phased_op)
        return phased_op
    circuit = transformer_primitives.map_operations(circuit, map_func).unfreeze(copy=False)
    circuit.append(dump_tracked_phase(qubit_phase.keys()))
    circuit.batch_replace(((m, op, g.on(*op.qubits)) for (m, op), g in phased_xz_replacements.items()))
    return transformer_primitives.unroll_circuit_op(circuit)