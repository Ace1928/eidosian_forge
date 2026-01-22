import cirq
import cirq_web
import pytest
def test_Operation3DSymbol_basic():
    wire_symbols = ['X']
    location_info = [{'row': 0, 'col': 0}]
    color_info = ['black']
    moment = 1
    symbol = cirq_web.circuits.symbols.Operation3DSymbol(wire_symbols, location_info, color_info, moment)
    actual = symbol.to_typescript()
    expected = {'wire_symbols': ['X'], 'location_info': [{'row': 0, 'col': 0}], 'color_info': ['black'], 'moment': 1}
    assert actual == expected