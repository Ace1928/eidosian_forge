import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
def strat_act_on_from_apply_decompose(val: Any, args: 'cirq.SimulationState', qubits: Sequence['cirq.Qid']) -> bool:
    if isinstance(val, ops.Gate):
        decomposed = protocols.decompose_once_with_qubits(val, qubits, flatten=False, default=None)
    else:
        decomposed = protocols.decompose_once(val, flatten=False, default=None)
    if decomposed is None:
        return NotImplemented
    all_ancilla: Set['cirq.Qid'] = set()
    for operation in ops.flatten_to_ops(decomposed):
        curr_ancilla = tuple((q for q in operation.qubits if q not in args.qubits))
        args = args.add_qubits(curr_ancilla)
        if args is NotImplemented:
            return NotImplemented
        all_ancilla.update(curr_ancilla)
        protocols.act_on(operation, args)
    args = args.remove_qubits(tuple(all_ancilla))
    if args is NotImplemented:
        raise TypeError(f'{type(args)} implements add_qubits but not remove_qubits.')
    return True