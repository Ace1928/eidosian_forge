from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
def make_transformer_func(add_deep_support: bool=False) -> cirq.TRANSFORMER:
    my_mock = mock.Mock()

    @cirq.transformer(add_deep_support=add_deep_support)
    def mock_tranformer_func(circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext]=None) -> cirq.Circuit:
        my_mock(circuit, context)
        return circuit.unfreeze()
    mock_tranformer_func.mock = my_mock
    return mock_tranformer_func