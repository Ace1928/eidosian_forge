from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
def get_circuit_inp_out_cirqsim(circuit: cirq.AbstractCircuit, qubit_order: Sequence[cirq.Qid], inputs: Sequence[int], outputs: Sequence[int], decimals: int=2) -> Tuple[str, str]:
    """Use a Cirq simulator to get a outputs of a `circuit`.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubit_order: The qubit order to pass to the cirq simulator.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.

    Returns:
        actual: The simulated output state as a string bitstring.
        should_be: The outputs argument formatted as a string bitstring for ease of comparison.
    """
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, initial_state=inputs, qubit_order=qubit_order)
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    should_be = ''.join((str(x) for x in outputs))
    return (actual, should_be)