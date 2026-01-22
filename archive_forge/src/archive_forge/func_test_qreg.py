import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
def test_qreg():
    lexer = QasmLexer()
    lexer.input('qreg [5];')
    token = lexer.token()
    assert token.type == 'QREG'
    assert token.value == 'qreg'
    token = lexer.token()
    assert token.type == '['
    assert token.value == '['
    token = lexer.token()
    assert token.type == 'NATURAL_NUMBER'
    assert token.value == 5
    token = lexer.token()
    assert token.type == ']'
    assert token.value == ']'
    token = lexer.token()
    assert token.type == ';'
    assert token.value == ';'