from qpd_test.tests_base import TestsBase
import pandas as pd
def test_join_semi(self):
    a = ([[0, 1], [None, 3]], ['a', 'b'])
    b = ([[0, 10], [None, 30]], ['a', 'b'])
    self.assert_eq(dict(a=a, b=b), 'SELECT * FROM a LEFT SEMI JOIN b ON a.a=b.a', [[0, 1]], ['a', 'b'])
    self.assert_eq(dict(a=a, b=b), 'SELECT a.* FROM a LEFT SEMI JOIN b ON a.a=b.a', [[0, 1]], ['a', 'b'])