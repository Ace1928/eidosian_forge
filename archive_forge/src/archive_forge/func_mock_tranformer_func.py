from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@cirq.transformer(add_deep_support=add_deep_support)
def mock_tranformer_func(circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext]=None) -> cirq.Circuit:
    my_mock(circuit, context)
    return circuit.unfreeze()