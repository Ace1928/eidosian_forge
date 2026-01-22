import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_damping_channel_text_diagram():
    pd = cirq.phase_damp(0.1000009)
    assert cirq.circuit_diagram_info(pd, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('PD(0.100001)',))
    assert cirq.circuit_diagram_info(pd, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('PD(0.1)',))
    assert cirq.circuit_diagram_info(pd, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('PD(0.1000009)',))