from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
def make_transformer_func_with_defaults() -> cirq.TRANSFORMER:
    my_mock = mock.Mock()

    @cirq.transformer
    def func(circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext]=cirq.TransformerContext(), atol: float=0.0001, custom_arg: CustomArg=CustomArg()) -> cirq.FrozenCircuit:
        my_mock(circuit, context, atol, custom_arg)
        return circuit.freeze()
    func.mock = my_mock
    return func