from yaql.language import exceptions
import yaql.tests
def test_dict_set_many_inline(self):
    data = {'a': 12, 'b c': 44}
    self.assertEqual({'a': 55, 'b c': 44, 'd x': 99}, self.eval('$.set(a => 55, "d x" => 99)', data=data))
    self.assertEqual(data, {'a': 12, 'b c': 44})