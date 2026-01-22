import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_channel_text_diagram():
    a = cirq.asymmetric_depolarize(1 / 9, 2 / 9, 3 / 9)
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('A(0.1111111111111111,0.2222222222222222,' + '0.3333333333333333)',))
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('A(0.111111,0.222222,0.333333)',))
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('A(0.11,0.22,0.33)',))