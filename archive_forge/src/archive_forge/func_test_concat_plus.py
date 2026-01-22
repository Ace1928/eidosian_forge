from yaql.language import exceptions
import yaql.tests
def test_concat_plus(self):
    self.assertEqual('abc', self.eval('a +b + c'))