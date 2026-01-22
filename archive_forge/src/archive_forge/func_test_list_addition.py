from yaql.language import exceptions
import yaql.tests
def test_list_addition(self):
    self.assertEqual([1, 2, 3, 4], self.eval('list(1, 2) + list(3, 4)'))
    self.assertEqual([1, 2, 6, 8], self.eval('list(1, 2) + list(3, 4).select($ * 2)'))