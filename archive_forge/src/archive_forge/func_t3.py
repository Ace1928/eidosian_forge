from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@cirq.transformer
def t3(circuit: cirq.AbstractCircuit, context: Optional[cirq.TransformerContext]=None) -> cirq.Circuit:
    assert context is not None
    context.logger.log('First INFO Log', 'of T3 Start')
    circuit = t1(circuit, context=context)
    context.logger.log('Second INFO Log', 'of T3 Middle')
    circuit = t2(circuit, context=context)
    context.logger.log('Third INFO Log', 'of T3 End')
    return circuit.unfreeze()