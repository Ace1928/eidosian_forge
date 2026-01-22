from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_call_method(self):
    self.assertEqual(2, self.eval('call(len, [], {}, [1,2])'))
    self.assertRaises(exceptions.NoMatchingMethodException, self.eval, 'call(len, [[1,2]], {}, [1,2])')
    self.assertRaises(exceptions.NoMatchingMethodException, self.eval, 'call(len, [], {sequence => [1,2]}, [1, 2])')
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'call(len, [], null, [1, 2])')
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'call(len, null, {sequence => [1,2]}, [1, 2])')
    self.assertTrue(self.eval('call(isEmpty, [], {}, null)'))