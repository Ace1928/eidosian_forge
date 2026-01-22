from yaql.language import exceptions
import yaql.tests
def test_keyword_dict_access(self):
    data = {'A': 12, 'b c': 44, '__d': 99, '_e': 999}
    self.assertEqual(12, self.eval('$.A', data=data))
    self.assertEqual(999, self.eval('$._e', data=data))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, "$.'b c'", data=data)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, '$.123', data=data)
    self.assertIsNone(self.eval('$.b', data=data))
    self.assertRaises(exceptions.YaqlLexicalException, self.eval, '$.__d', data=data)