import sys
import unittest
from Cython.Utils import (
def test_try_finally_contextmanager(self):
    states = []

    @try_finally_contextmanager
    def gen(*args, **kwargs):
        states.append('enter')
        yield (args, kwargs)
        states.append('exit')
    with gen(1, 2, 3, x=4) as call_args:
        assert states == ['enter']
        self.assertEqual(call_args, ((1, 2, 3), {'x': 4}))
    assert states == ['enter', 'exit']

    class MyException(RuntimeError):
        pass
    del states[:]
    with self.assertRaises(MyException):
        with gen(1, 2, y=4) as call_args:
            assert states == ['enter']
            self.assertEqual(call_args, ((1, 2), {'y': 4}))
            raise MyException('FAIL INSIDE')
        assert states == ['enter', 'exit']
    del states[:]
    with self.assertRaises(StopIteration):
        with gen(1, 2, y=4) as call_args:
            assert states == ['enter']
            self.assertEqual(call_args, ((1, 2), {'y': 4}))
            raise StopIteration('STOP')
        assert states == ['enter', 'exit']