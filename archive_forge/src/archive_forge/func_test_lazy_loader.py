from cirq import _import
def test_lazy_loader():
    linalg = _import.LazyLoader('linalg', globals(), 'scipy.linalg')
    linalg.fun = 1
    assert linalg._module is None
    assert 'linalg' not in linalg.__dict__
    linalg.det([[1]])
    assert linalg._module is not None
    assert globals()['linalg'] == linalg._module
    assert 'fun' in linalg.__dict__
    assert 'LinAlgError' in dir(linalg)
    assert linalg.fun == 1