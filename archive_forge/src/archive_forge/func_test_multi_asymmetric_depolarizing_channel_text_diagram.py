import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_multi_asymmetric_depolarizing_channel_text_diagram():
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('A(II:0.6666666666666666, XX:0.3333333333333333)', '(1)'))
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('A(II:0.666667, XX:0.333333)', '(1)'))
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('A(II:0.67, XX:0.33)', '(1)'))
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('A(II:0.6666666666666666, XX:0.3333333333333333)', '(1)'))