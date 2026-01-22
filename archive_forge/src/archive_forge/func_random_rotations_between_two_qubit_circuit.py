import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def random_rotations_between_two_qubit_circuit(q0: 'cirq.Qid', q1: 'cirq.Qid', depth: int, two_qubit_op_factory: Callable[['cirq.Qid', 'cirq.Qid', 'np.random.RandomState'], 'cirq.OP_TREE']=lambda a, b, _: ops.CZPowGate()(a, b), single_qubit_gates: Sequence['cirq.Gate']=(ops.X ** 0.5, ops.Y ** 0.5, ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)), add_final_single_qubit_layer: bool=True, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> 'cirq.Circuit':
    """Generate a random two-qubit quantum circuit.

    This construction uses a similar structure to those in the paper
    https://www.nature.com/articles/s41586-019-1666-5.

    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates,
    controlled by their respective arguments, see below.

    Args:
        q0: The first qubit
        q1: The second qubit
        depth: The number of cycles.
        two_qubit_op_factory: A callable that returns a two-qubit operation.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(q0, q1, prng)`, where `prng` is the
            pseudorandom number generator.
        single_qubit_gates: Single-qubit gates are selected randomly from this
            sequence. No qubit is acted upon by the same single-qubit gate in
            consecutive cycles. If only one choice of single-qubit gate is
            given, then this constraint is not enforced.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle (subject to the same
            non-consecutivity constraint).
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = value.parse_random_state(seed)
    circuit = circuits.Circuit()
    previous_single_qubit_layer = circuits.Moment()
    single_qubit_layer_factory = _single_qubit_gates_arg_to_factory(single_qubit_gates=single_qubit_gates, qubits=(q0, q1), prng=prng)
    for _ in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
        circuit += single_qubit_layer
        circuit += two_qubit_op_factory(q0, q1, prng)
        previous_single_qubit_layer = single_qubit_layer
    if add_final_single_qubit_layer:
        circuit += single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
    return circuit