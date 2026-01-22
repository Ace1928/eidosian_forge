import yaql.tests
def test_unary_plus(self):
    self.assertEqual(4, self.eval('+4'))
    self.assertEqual(12.0, self.eval('+12.0'))
    self.assertEqual(2, self.eval('3-+1'))
    self.assertEqual(4, self.eval('3++1'))
    self.assertAlmostEqual(2.1, self.eval('3.2 - +1.1'))