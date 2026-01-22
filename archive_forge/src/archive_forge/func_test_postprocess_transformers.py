import pytest
import sympy
import cirq
from cirq_aqt import aqt_target_gateset
def test_postprocess_transformers():
    gs = aqt_target_gateset.AQTTargetGateset()
    assert len(gs.postprocess_transformers) == 2