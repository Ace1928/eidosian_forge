import yaql.tests
def test_not_matches_operator_string(self):
    self.assertFalse(self.eval("axb !~ 'a.b'"))
    self.assertTrue(self.eval("abx !~ 'a.b'"))