import dill
from functools import partial
import warnings
def test_super():
    assert copy(obj1(), byref=True)
    assert copy(obj1(), byref=True, recurse=True)
    assert copy(obj1(), recurse=True)
    assert copy(obj1())
    assert copy(obj2(), byref=True)
    assert copy(obj2(), byref=True, recurse=True)
    assert copy(obj2(), recurse=True)
    assert copy(obj2())
    assert copy(obj3(), byref=True)
    assert copy(obj3(), byref=True, recurse=True)
    assert copy(obj3(), recurse=True)
    assert copy(obj3())