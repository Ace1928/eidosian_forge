import unittest
from numba.tests.support import captured_stdout
def test_ex_nested_list(self):
    with captured_stdout():
        from numba.typed import List
        mylist = List()
        for i in range(10):
            l = List()
            for i in range(10):
                l.append(i)
            mylist.append(l)
        print(mylist)