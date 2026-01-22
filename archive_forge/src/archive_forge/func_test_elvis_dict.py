from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_elvis_dict(self):
    self.assertEqual(1, self.eval('$?.a', data={'a': 1}))
    self.assertIsNone(self.eval('$?.a', data=None))