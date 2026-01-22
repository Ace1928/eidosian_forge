import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_nocompile_context_leaves_invalid_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a').with_tags('nocompile'), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    with pytest.raises(ValueError, match='Deferred measurement for key=a not found'):
        _ = cirq.defer_measurements(circuit, context=cirq.TransformerContext(tags_to_ignore=('nocompile',)))