import cirq.testing
def test_fake_printer():
    p = cirq.testing.FakePrinter()
    assert p.text_pretty == ''
    p.text('stuff')
    assert p.text_pretty == 'stuff'
    p.text(' more')
    assert p.text_pretty == 'stuff more'