from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_lambda_closure(self):
    data = [1, 2, 3, 4, 5, 6]
    self.assertEqual([3, 4, 5, 6], self.eval('$.where(lambda($ > 3)($+1))', data=data))
    self.assertEqual([4, 5, 6], self.eval('$.where(lambda($ > 3)())', data=data))