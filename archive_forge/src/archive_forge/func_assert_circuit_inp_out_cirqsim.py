from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
def assert_circuit_inp_out_cirqsim(circuit: cirq.AbstractCircuit, qubit_order: Sequence[cirq.Qid], inputs: Sequence[int], outputs: Sequence[int], decimals: int=2):
    """Use a Cirq simulator to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubit_order: The qubit order to pass to the cirq simulator.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.
    """
    actual, should_be = get_circuit_inp_out_cirqsim(circuit, qubit_order, inputs, outputs, decimals)
    assert actual == should_be