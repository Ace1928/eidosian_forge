import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def sample_measurement_ops(self, measurement_ops: List['cirq.GateOperation'], repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, *, _allow_repeated=False) -> Dict[str, np.ndarray]:
    """Samples from the system at this point in the computation.

        Note that this does not collapse the state vector.

        In contrast to `sample` which samples qubits, this takes a list of
        `cirq.GateOperation` instances whose gates are `cirq.MeasurementGate`
        instances and then returns a mapping from the key in the measurement
        gate to the resulting bit strings. Different measurement operations must
        not act on the same qubits.

        Args:
            measurement_ops: `GateOperation` instances whose gates are
                `MeasurementGate` instances to be sampled form.
            repetitions: The number of samples to take.
            seed: A seed for the pseudorandom number generator.
            _allow_repeated: If True, adds extra dimension to the result,
                corresponding to the number of times a key is repeated.

        Returns: A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)

        Raises:
            ValueError: If the operation's gates are not `MeasurementGate`
                instances or a qubit is acted upon multiple times by different
                operations from `measurement_ops`.
        """
    for op in measurement_ops:
        gate = op.gate
        if not isinstance(gate, ops.MeasurementGate):
            raise ValueError(f'{op.gate} was not a MeasurementGate')
    result = collections.Counter((key for op in measurement_ops for key in protocols.measurement_key_names(op)))
    if result and (not _allow_repeated):
        duplicates = [k for k, v in result.most_common() if v > 1]
        if duplicates:
            raise ValueError(f'Measurement key {','.join(duplicates)} repeated')
    measured_qubits = []
    seen_qubits: Set[cirq.Qid] = set()
    for op in measurement_ops:
        for q in op.qubits:
            if q not in seen_qubits:
                seen_qubits.add(q)
                measured_qubits.append(q)
    indexed_sample = self.sample(measured_qubits, repetitions, seed=seed)
    results: Dict[str, Any] = {}
    qubits_to_index = {q: i for i, q in enumerate(measured_qubits)}
    for op in measurement_ops:
        gate = cast(ops.MeasurementGate, op.gate)
        key = gate.key
        out = np.zeros(shape=(repetitions, len(op.qubits)), dtype=np.int8)
        inv_mask = gate.full_invert_mask()
        cmap = gate.confusion_map
        for i, q in enumerate(op.qubits):
            out[:, i] = indexed_sample[:, qubits_to_index[q]]
            if inv_mask[i]:
                out[:, i] ^= out[:, i] < 2
        self._confuse_results(out, op.qubits, cmap, seed)
        if _allow_repeated:
            if key not in results:
                results[key] = []
            results[key].append(out)
        else:
            results[gate.key] = out
    return results if not _allow_repeated else {k: np.array(v).swapaxes(0, 1) for k, v in results.items()}