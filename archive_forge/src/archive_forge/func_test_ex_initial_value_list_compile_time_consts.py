import unittest
from numba.tests.support import captured_stdout
from numba import typed
def test_ex_initial_value_list_compile_time_consts(self):
    with captured_stdout():
        from numba import njit, literally
        from numba.extending import overload

        def specialize(x):
            pass

        @overload(specialize)
        def ol_specialize(x):
            iv = x.initial_value
            if iv is None:
                return lambda x: literally(x)
            assert iv == [1, 2, 3]
            return lambda x: x

        @njit
        def foo():
            l = [1, 2, 3]
            l[2] = 20
            l.append(30)
            return specialize(l)
        result = foo()
        print(result)
    expected = [1, 2, 20, 30]
    self.assertEqual(result, expected)