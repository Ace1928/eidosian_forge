import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
@pytest.mark.parametrize('number', ['00000', '03', '3', '0045', '21'])
def test_natural_numbers(number: str):
    lexer = QasmLexer()
    lexer.input(number)
    token = lexer.token()
    assert token is not None
    assert token.type == 'NATURAL_NUMBER'
    assert token.value == int(number)