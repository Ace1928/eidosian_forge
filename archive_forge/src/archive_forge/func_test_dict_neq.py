from yaql.language import exceptions
import yaql.tests
def test_dict_neq(self):
    self.assertFalse(self.eval('{a => [c, 55]} != {a => [c, 55]}'))
    self.assertFalse(self.eval('{[c, 55] => a} != {[c, 55] => a}'))
    self.assertTrue(self.eval('{[c, 55] => a} != {[c, 56] => a}'))
    self.assertTrue(self.eval('{[c, 55] => a, b => 1} != {[c, 55] => a}'))
    self.assertTrue(self.eval('{[c, 55] => a} != null'))