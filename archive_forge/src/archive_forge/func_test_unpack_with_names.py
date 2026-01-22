from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_unpack_with_names(self):
    self.assertEqual(5, self.eval('[2, 3].unpack(a, b) -> $a + $b'))
    self.assertRaises(ValueError, self.eval, '[2, 3].unpack(a, b, c) -> $a + $b')
    self.assertRaises(ValueError, self.eval, '[2, 3].unpack(a) -> $a')