import unittest
from numba.tests.support import captured_stdout
def test_ex_inferred_list(self):
    with captured_stdout():
        from numba import njit
        from numba.typed import List

        @njit
        def foo(mylist):
            for i in range(10, 20):
                mylist.append(i)
            return mylist
        l = List()
        l.append(42)
        print(l[0])
        l[0] = 23
        print(l[0])
        print(len(l))
        l.pop()
        print(len(l))
        l = foo(l)
        print(len(l))
        py_list = [2, 3, 5]
        numba_list = List(py_list)
        print(len(numba_list))