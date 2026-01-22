from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
def test_closure_in_escaper(self):

    def impl_factory(consumer_func):

        def impl():

            def callinner():

                def inner():
                    return 10
                return inner()
            return consumer_func(callinner)
        return impl
    cfunc = njit(impl_factory(consumer))
    impl = impl_factory(consumer.py_func)
    self.assertEqual(impl(), cfunc())