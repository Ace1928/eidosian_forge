from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_create_transformer_with_kwargs_raises():
    with pytest.raises(SyntaxError, match='must not contain `context`'):
        cirq.create_transformer_with_kwargs(cirq.merge_k_qubit_unitaries, k=2, context=cirq.TransformerContext())