import numpy as np
from xarray import DataArray, Dataset, Variable
def test_dataarray_typed_ops() -> None:
    """Tests for type checking of typed_ops on DataArray"""
    da = DataArray([1, 2, 3], dims=['t'])

    def _test(da: DataArray) -> None:
        assert isinstance(da, DataArray)
    _int: int = 1
    _list = [1, 2, 3]
    _ndarray = np.array([1, 2, 3])
    _var = Variable(dims=['t'], data=[1, 2, 3])
    _test(da + _int)
    _test(da + _list)
    _test(da + _ndarray)
    _test(da + _var)
    _test(da + da)
    _test(_int + da)
    _test(_list + da)
    _test(_ndarray + da)
    _test(_var + da)
    _test(da == _int)
    _test(da == _list)
    _test(da == _ndarray)
    _test(da == _var)
    _test(_int == da)
    _test(_list == da)
    _test(_ndarray == da)
    _test(_var == da)
    _test(da < _int)
    _test(da < _list)
    _test(da < _ndarray)
    _test(da < _var)
    _test(_int > da)
    _test(_list > da)
    _test(_ndarray > da)
    _test(_var > da)
    da += _int
    da += _list
    da += _ndarray
    da += _var
    _test(-da)