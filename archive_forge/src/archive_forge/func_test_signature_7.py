from numba.np.ufunc.deviceufunc import GUFuncEngine
import unittest
def test_signature_7(self):
    signature = '(), () -> ()'
    shapes = ((5,), ())
    expects = dict(ishapes=[(), ()], oshapes=[()], loopdims=(5,), pinned=[False, True])
    template(signature, shapes, expects)