from yaql.language import exceptions
import yaql.tests
def test_concat_func(self):
    self.assertEqual('abc', self.eval('concat(a, b, c)'))