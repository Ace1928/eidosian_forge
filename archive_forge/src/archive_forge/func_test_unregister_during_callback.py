import unittest
from unittest.mock import Mock
from IPython.core import events
import IPython.testing.tools as tt
def test_unregister_during_callback(self):
    invoked = [False] * 3

    def func1(*_):
        invoked[0] = True
        self.em.unregister('ping_received', func1)
        self.em.register('ping_received', func3)

    def func2(*_):
        invoked[1] = True
        self.em.unregister('ping_received', func2)

    def func3(*_):
        invoked[2] = True
    self.em.register('ping_received', func1)
    self.em.register('ping_received', func2)
    self.em.trigger('ping_received')
    self.assertEqual([True, True, False], invoked)
    self.assertEqual([func3], self.em.callbacks['ping_received'])