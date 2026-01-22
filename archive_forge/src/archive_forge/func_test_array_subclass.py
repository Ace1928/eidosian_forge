import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_array_subclass():
    try:
        import numpy as np

        class TestArray(np.ndarray):

            def __new__(cls, input_array, color):
                obj = np.asarray(input_array).view(cls)
                obj.color = color
                return obj

            def __array_finalize__(self, obj):
                if obj is None:
                    return
                if isinstance(obj, type(self)):
                    self.color = obj.color

            def __getnewargs__(self):
                return (np.asarray(self), self.color)
        a1 = TestArray(np.zeros(100), color='green')
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a1)
            assert a1.__dict__ == dill.copy(a1).__dict__
        a2 = a1[0:9]
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a2)
            assert a2.__dict__ == dill.copy(a2).__dict__

        class TestArray2(np.ndarray):
            color = 'blue'
        a3 = TestArray2([1, 2, 3, 4, 5])
        a3.color = 'green'
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a3)
            assert a3.__dict__ == dill.copy(a3).__dict__
    except ImportError:
        pass