from yaql.language import exceptions
import yaql.tests
def test_indexer_dict_access_with(self):
    data = {'a': 12, 'b c': 44}
    self.assertEqual(55, self.eval('$[c, 55]', data=data))
    self.assertEqual(66, self.eval('$[c, default => 66]', data=data))