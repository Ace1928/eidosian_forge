import yaql.tests
def test_boolean_equality(self):
    self.assertTrue(self.eval('false = false'))
    self.assertTrue(self.eval('false != true'))
    self.assertTrue(self.eval('true != false'))
    self.assertTrue(self.eval('true = true'))
    self.assertFalse(self.eval('true = false'))
    self.assertFalse(self.eval('false = true'))
    self.assertFalse(self.eval('false != false'))
    self.assertFalse(self.eval('true != true'))