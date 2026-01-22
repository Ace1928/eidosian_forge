import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
def measure_confusion_matrix(sampler: 'cirq.Sampler', qubits: Union[Sequence['cirq.Qid'], Sequence[Sequence['cirq.Qid']]], repetitions: int=1000) -> TensoredConfusionMatrices:
    """Prepares `TensoredConfusionMatrices` for the n qubits in the input.

    The confusion matrix (CM) for two qubits is the following matrix:

        ⎡ Pr(00|00) Pr(01|00) Pr(10|00) Pr(11|00) ⎤
        ⎢ Pr(00|01) Pr(01|01) Pr(10|01) Pr(11|01) ⎥
        ⎢ Pr(00|10) Pr(01|10) Pr(10|10) Pr(11|10) ⎥
        ⎣ Pr(00|11) Pr(01|11) Pr(10|11) Pr(11|11) ⎦

    where Pr(ij | pq) = Probability of observing “ij” given state “pq” was prepared.

    Args:
        sampler: Sampler to collect the data from.
        qubits: Qubits for which the confusion matrix should be measured.
        repetitions: Number of times to sample each circuit for a confusion matrix row.
    """
    qubits = cast(Sequence[Sequence['cirq.Qid']], [qubits] if isinstance(qubits[0], ops.Qid) else qubits)
    confusion_matrices = []
    for qs in qubits:
        flip_symbols = sympy.symbols(f'flip_0:{len(qs)}')
        flip_circuit = circuits.Circuit([ops.X(q) ** s for q, s in zip(qs, flip_symbols)], ops.measure(*qs))
        sweeps = study.Product(*[study.Points(f'flip_{i}', [0, 1]) for i in range(len(qs))])
        results = sampler.run_sweep(flip_circuit, sweeps, repetitions=repetitions)
        confusion_matrices.append(np.asarray([vis.get_state_histogram(r) for r in results], dtype=float) / repetitions)
    return TensoredConfusionMatrices(confusion_matrices, qubits, repetitions=repetitions, timestamp=time.time())