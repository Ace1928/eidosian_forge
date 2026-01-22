from yaql.language import exceptions
import yaql.tests
def test_dict_set_many(self):
    data = {'a': 12, 'b c': 44}
    self.assertEqual({'a': 55, 'b c': 44, 'd x': 99, None: None}, self.eval('$.set(dict(a => 55, "d x" => 99, null => null))', data=data))
    self.assertEqual(data, {'a': 12, 'b c': 44})