from yaql.language import exceptions
import yaql.tests
def test_dict_dict_key(self):
    self.assertEqual(3, self.eval('dict($ => 3).get($)', data={'a': 1}))