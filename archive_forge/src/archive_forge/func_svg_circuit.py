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
def svg_circuit(circuit: 'cirq.AbstractCircuit', registers: Iterable[gate_with_registers.Register]=(), include_costs: bool=False):
    """Return an SVG object representing a circuit.

    Args:
        circuit: The circuit to draw.
        registers: Optional `Signature` object to order the qubits.
        include_costs: If true, each operation is annotated with it's T-complexity cost.

    Raises:
        ValueError: If `circuit` is empty.
    """
    if len(circuit) == 0:
        raise ValueError('Circuit is empty.')
    if registers:
        qubit_order = cirq.QubitOrder.explicit(merge_qubits(registers, **get_named_qubits(registers)), fallback=cirq.QubitOrder.DEFAULT)
    else:
        qubit_order = cirq.QubitOrder.DEFAULT
    if include_costs:
        circuit = circuit_with_costs(circuit)
    tdd = circuit.to_text_diagram_drawer(transpose=False, qubit_order=qubit_order)
    if len(tdd.horizontal_lines) == 0:
        raise ValueError('No non-empty moments.')
    return IPython.display.SVG(ccsvg.tdd_to_svg(tdd))