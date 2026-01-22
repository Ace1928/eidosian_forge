from yaql.language import exceptions
import yaql.tests
def test_indexer_list_access(self):
    data = [1, 2, 3]
    self.assertEqual(1, self.eval('$[0]', data=data))
    self.assertEqual(3, self.eval('$[-1]', data=data))
    self.assertEqual(2, self.eval('$[-1-1]', data=data))
    self.assertRaises(IndexError, self.eval, '$[3]', data=data)
    self.assertRaises(IndexError, self.eval, '$[-4]', data=data)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, '$[a]', data=data)