import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
def test_deprecated_classmethod(self):

    class A:

        def __init__(self):
            pass

        @classmethod
        def classmethod_a(self):
            pass

        @classmethod
        @deprecated(classmethod_a)
        def classmethod_b(self):
            return 'b'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert A().classmethod_b() == 'b'
        assert issubclass(w[-1].category, FutureWarning)

    class A:

        def __init__(self):
            pass

        @classmethod
        def classmethod_a(self):
            pass

        @classmethod
        @deprecated(classmethod_a, category=DeprecationWarning)
        def classmethod_b(self):
            return 'b'
    with pytest.warns(DeprecationWarning):
        assert A().classmethod_b() == 'b'