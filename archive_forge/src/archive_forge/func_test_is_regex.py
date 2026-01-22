import yaql.tests
def test_is_regex(self):
    self.assertTrue(self.eval('isRegex(regex("a.b"))'))
    self.assertFalse(self.eval('isRegex(123)'))
    self.assertFalse(self.eval('isRegex(abc)'))