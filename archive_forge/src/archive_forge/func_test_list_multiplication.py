from yaql.language import exceptions
import yaql.tests
def test_list_multiplication(self):
    self.assertCountEqual([1, 2, 1, 2, 1, 2], self.eval('3 * [1, 2]'))
    self.assertCountEqual([1, 2, 1, 2, 1, 2], self.eval('[1, 2] * 3'))