from qpd_test.tests_base import TestsBase
import pandas as pd
def test_order_by_no_limit(self):
    a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=float)
    self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a ORDER BY a\n                ', a=a)