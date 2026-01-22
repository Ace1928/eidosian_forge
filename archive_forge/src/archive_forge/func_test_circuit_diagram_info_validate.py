import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_validate():
    with pytest.raises(ValueError):
        _ = cirq.CircuitDiagramInfo('X')