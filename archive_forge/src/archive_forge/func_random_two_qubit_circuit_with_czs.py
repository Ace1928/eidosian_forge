from typing import List, Union, Sequence, Dict, Optional, TYPE_CHECKING
from cirq import circuits, ops, value
from cirq.ops import Qid
from cirq._doc import document
def random_two_qubit_circuit_with_czs(num_czs: int=3, q0: Optional[Qid]=None, q1: Optional[Qid]=None, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> circuits.Circuit:
    """Creates a random two qubit circuit with the given number of CNOTs.

    The resulting circuit will have `num_cnots` number of CNOTs that will be
    surrounded by random `PhasedXPowGate` instances on both qubits.

    Args:
         num_czs: the number of CNOTs to be guaranteed in the circuit
         q0: the first qubit the circuit should operate on
         q1: the second qubit the circuit should operate on
         random_state: an optional random seed
    Returns:
         the random two qubit circuit
    """
    prng = value.parse_random_state(random_state)
    q0 = ops.NamedQubit('q0') if q0 is None else q0
    q1 = ops.NamedQubit('q1') if q1 is None else q1

    def random_one_qubit_gate():
        return ops.PhasedXPowGate(phase_exponent=prng.rand(), exponent=prng.rand())

    def one_cz():
        return [ops.CZ.on(q0, q1), random_one_qubit_gate().on(q0), random_one_qubit_gate().on(q1)]
    return circuits.Circuit([random_one_qubit_gate().on(q0), random_one_qubit_gate().on(q1), [one_cz() for _ in range(num_czs)]])