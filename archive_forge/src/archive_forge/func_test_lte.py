import yaql.tests
def test_lte(self):
    res = self.eval('3 <= 5')
    self.assertIsInstance(res, bool)
    self.assertTrue(res)
    self.assertTrue(self.eval('3 <= 3'))
    self.assertFalse(self.eval('3 <= 2'))