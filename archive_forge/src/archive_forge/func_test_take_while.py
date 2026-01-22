from yaql.language import exceptions
import yaql.tests
def test_take_while(self):
    self.assertEqual([1, 2], self.eval('[1, 2, 3, 4].takeWhile($ < 3)'))
    self.assertEqual([1, 2], self.eval('takeWhile([1, 2, 3, 4], $ < 3)'))