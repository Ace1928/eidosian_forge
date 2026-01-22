import yaql.tests
def test_multiplication_int(self):
    res = self.eval('3 * 2')
    self.assertEqual(6, res)
    self.assertIsInstance(res, int)
    self.assertEqual(-6, self.eval('3 * -2'))
    self.assertEqual(6, self.eval('-3 * -2'))