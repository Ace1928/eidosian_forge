from yaql.language import exceptions
import yaql.tests
def test_dict_list_key(self):
    self.assertEqual(3, self.eval('dict($ => 3).get($)', data=[1, 2]))
    self.assertEqual(3, self.eval('dict($ => 3).get($)', data=[1, [2]]))