from qpd_test.tests_base import TestsBase
import pandas as pd
def test_join_inner_special_chars(self):
    a = self.make_rand_df(100, **{'a b': (int, 40), 'b': (str, 40), 'c': (float, 40)})
    b = self.make_rand_df(80, **{'d': (float, 10), 'a b': (int, 10), 'b': (str, 10)})
    self.eq_sqlite('SELECT a.*, d, d*c AS x FROM a INNER JOIN b ON a.`a b`=b.`a b` AND a.b=b.b', a=a, b=b)