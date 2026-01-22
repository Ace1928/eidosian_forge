from pathlib import Path
from typing import Iterable
import cirq
import cirq.contrib.svg.svg as ccsvg
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import nbformat
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, get_named_qubits, merge_qubits
from nbconvert.preprocessors import ExecutePreprocessor
def circuit_with_costs(circuit: 'cirq.AbstractCircuit') -> 'cirq.AbstractCircuit':
    """Annotates each operation in the circuit with its T-complexity cost."""

    def _map_func(op: cirq.Operation, _):
        t_cost = t_complexity_protocol.t_complexity(op)
        return op.with_tags(f't:{t_cost.t:g},r:{t_cost.rotations:g}')
    return cirq.map_operations(circuit, map_func=_map_func)