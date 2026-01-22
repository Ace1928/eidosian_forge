from yaql.language import exceptions
import yaql.tests
def test_list_expr(self):
    self.assertEqual([1, 2, 3], self.eval('[1,2,3]'))
    self.assertEqual([2, 4], self.eval('[1,[2]][1] + [3, [4]][1]'))
    self.assertEqual([1, 2, 3, 4], self.eval('[1,2] + [3, 4]'))
    self.assertEqual(2, self.eval('([1,2] + [3, 4])[1]'))
    self.assertEqual([], self.eval('[]'))