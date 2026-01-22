import types
import testtools
from fixtures.callmany import CallMany
def test__call__raise_errors_false_callsall_returns_exceptions(self):
    calls = []

    def raise_exception1():
        calls.append('1')
        raise Exception('woo')

    def raise_exception2():
        calls.append('2')
        raise Exception('woo')
    call = CallMany()
    call.push(raise_exception2)
    call.push(raise_exception1)
    exceptions = call(raise_errors=False)
    self.assertEqual(['1', '2'], calls)
    self.assertEqual(2, len(exceptions))
    self.assertEqual(3, len(exceptions[0]))
    type, value, tb = exceptions[0]
    self.assertEqual(Exception, type)
    self.assertIsInstance(value, Exception)
    self.assertEqual(('woo',), value.args)
    self.assertIsInstance(tb, types.TracebackType)