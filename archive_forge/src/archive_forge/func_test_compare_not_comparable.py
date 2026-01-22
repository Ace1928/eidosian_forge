from yaql.language import exceptions
import yaql.tests
def test_compare_not_comparable(self):
    self.assertTrue(self.eval('asd != true'))
    self.assertFalse(self.eval('asd = 0'))