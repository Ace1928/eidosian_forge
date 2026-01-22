import yaql.tests
def test_select_all_cases(self):
    expr = 'selectAllCases($ < 10, $ > 5)'
    self.assertEqual([0], self.eval(expr, data=1))
    self.assertEqual([0, 1], self.eval(expr, data=7))
    self.assertEqual([1], self.eval(expr, data=12))