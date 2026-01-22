from cirq.linalg.tolerance import all_near_zero, all_near_zero_mod, near_zero, near_zero_mod
def test_near_zero_mod():
    atol = 5
    assert near_zero_mod(0, 100, atol=atol)
    assert near_zero_mod(4.5, 100, atol=atol)
    assert not near_zero_mod(5.5, 100, atol=atol)
    assert near_zero_mod(100, 100, atol=atol)
    assert near_zero_mod(95.5, 100, atol=atol)
    assert not near_zero_mod(94.5, 100, atol=atol)
    assert near_zero_mod(-4.5, 100, atol=atol)
    assert not near_zero_mod(-5.5, 100, atol=atol)
    assert near_zero_mod(104.5, 100, atol=atol)
    assert not near_zero_mod(105.5, 100, atol=atol)