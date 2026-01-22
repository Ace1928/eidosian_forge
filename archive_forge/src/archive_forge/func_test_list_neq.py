from yaql.language import exceptions
import yaql.tests
def test_list_neq(self):
    self.assertFalse(self.eval('[c, 55] != [c, 55]'))
    self.assertTrue(self.eval('[c, 55] != [55, c]'))
    self.assertTrue(self.eval('[c, 55] != null'))
    self.assertTrue(self.eval('null != [c, 55]'))