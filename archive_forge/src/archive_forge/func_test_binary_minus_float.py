import yaql.tests
def test_binary_minus_float(self):
    res = self.eval('1 - 2.1')
    self.assertEqual(-1.1, res)
    self.assertIsInstance(res, float)
    res = self.eval('123.321- 0.321')
    self.assertEqual(123.0, res)
    self.assertIsInstance(res, float)