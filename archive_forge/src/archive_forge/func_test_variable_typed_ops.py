import numpy as np
from xarray import DataArray, Dataset, Variable
def test_variable_typed_ops() -> None:
    """Tests for type checking of typed_ops on Variable"""
    var = Variable(dims=['t'], data=[1, 2, 3])

    def _test(var: Variable) -> None:
        assert isinstance(var, Variable)
    _int: int = 1
    _list = [1, 2, 3]
    _ndarray = np.array([1, 2, 3])
    _test(var + _int)
    _test(var + _list)
    _test(var + _ndarray)
    _test(var + var)
    _test(_int + var)
    _test(_list + var)
    _test(_ndarray + var)
    _test(var == _int)
    _test(var == _list)
    _test(var == _ndarray)
    _test(_int == var)
    _test(_list == var)
    _test(_ndarray == var)
    _test(var < _int)
    _test(var < _list)
    _test(var < _ndarray)
    _test(_int > var)
    _test(_list > var)
    _test(_ndarray > var)
    var += _int
    var += _list
    var += _ndarray
    _test(-var)