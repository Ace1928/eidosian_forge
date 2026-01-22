from stevedore.example2 import fields
from stevedore.tests import utils
def test_long_item(self):
    f = fields.FieldList(25)
    text = ''.join(f.format({'name': 'a value longer than the allowed width'}))
    expected = '\n'.join([': name : a value longer', '    than the allowed', '    width', ''])
    self.assertEqual(text, expected)