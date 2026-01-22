import cirq
def test_comparsion_key():
    q = cirq.testing.NoIdentifierQubit()
    p = cirq.testing.NoIdentifierQubit()
    assert p == q