import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_method_call_whitelist_string(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj, whitelist=['m_foo'])
    self.assertEqual(3, self.eval('$.m_foo(5, 2)', obj))
    self.assertRaises(AttributeError, self.eval, '$.bar(a)', obj)