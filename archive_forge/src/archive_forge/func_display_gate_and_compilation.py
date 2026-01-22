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
def display_gate_and_compilation(g: cq_testing.GateHelper, vertical=False, include_costs=True):
    """Use ipywidgets to display SVG circuits for a `GateHelper` next to each other.

    Args:
        g: The `GateHelper` to draw
        vertical: If true, lay-out the original gate and its decomposition vertically
            rather than side-by-side.
        include_costs: If true, each operation is annotated with it's T-complexity cost.
    """
    out1 = ipywidgets.Output()
    out2 = ipywidgets.Output()
    if vertical:
        box = ipywidgets.VBox([out1, out2])
    else:
        box = ipywidgets.HBox([out1, out2])
    out1.append_display_data(svg_circuit(g.circuit, registers=g.r, include_costs=include_costs))
    out2.append_display_data(svg_circuit(cirq.Circuit(cirq.decompose_once(g.operation)), registers=g.r, include_costs=include_costs))
    IPython.display.display(box)