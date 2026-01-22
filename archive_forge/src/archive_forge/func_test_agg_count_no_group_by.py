from qpd_test.tests_base import TestsBase
import pandas as pd
def test_agg_count_no_group_by(self):
    a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
    self.eq_sqlite('\n                SELECT\n                    COUNT(a) AS c_a,\n                    COUNT(DISTINCT a) AS cd_a,\n                    COUNT(b) AS c_b,\n                    COUNT(DISTINCT b) AS cd_b,\n                    COUNT(c) AS c_c,\n                    COUNT(DISTINCT c) AS cd_c,\n                    COUNT(d) AS c_d,\n                    COUNT(DISTINCT d) AS cd_d,\n                    COUNT(e) AS c_e,\n                    COUNT(DISTINCT a) AS cd_e\n                FROM a\n                ', a=a)
    b = ([[1, 'x', 1.5], [2, None, 2.5], [2, None, 2.5]], ['a', 'b', 'c'])
    self.assert_eq(dict(a=a, b=b), '\n                SELECT\n                    COUNT(*) AS a1,\n                    COUNT(DISTINCT *) AS a2,\n                    COUNT(a, b) AS a3,\n                    COUNT(DISTINCT a,b) AS a4,\n                    COUNT(a, b) + COUNT(DISTINCT a,b) AS a5\n                FROM b\n                ', [[3, 2, 3, 2, 5]], ['a1', 'a2', 'a3', 'a4', 'a5'])