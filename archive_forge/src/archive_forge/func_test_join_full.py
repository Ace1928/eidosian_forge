from qpd_test.tests_base import TestsBase
import pandas as pd
def test_join_full(self):
    a = ([[0, 1], [None, 3]], ['a', 'b'])
    b = ([[0, 10], [None, 30]], ['a', 'c'])
    self.assert_eq(dict(a=a, b=b), 'SELECT a.*,c FROM a FULL JOIN b ON a.a=b.a', [[0, 1, 10], [None, 3, None], [None, None, 30]], ['a', 'b', 'c'])