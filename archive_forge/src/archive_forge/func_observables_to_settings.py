import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
def observables_to_settings(observables: Iterable['cirq.PauliString'], qubits: Iterable['cirq.Qid']) -> Iterable[InitObsSetting]:
    """Transform an observable to an InitObsSetting initialized in the
    all-zeros state.
    """
    for observable in observables:
        yield InitObsSetting(init_state=zeros_state(qubits), observable=observable)