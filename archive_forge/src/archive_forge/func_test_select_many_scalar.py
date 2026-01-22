from yaql.language import exceptions
import yaql.tests
def test_select_many_scalar(self):
    self.assertEqual(['xx', 'xx'], self.eval('range(2).selectMany(xx)'))