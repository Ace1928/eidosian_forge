import unittest
from unittest.mock import Mock
from IPython.core import events
import IPython.testing.tools as tt
def test_cb_keyboard_interrupt(self):
    cb = Mock(side_effect=KeyboardInterrupt)
    self.em.register('ping_received', cb)
    with tt.AssertPrints('Error in callback'):
        self.em.trigger('ping_received')