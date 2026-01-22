from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_listener_handler_for_function(self):

    def square(value):
        return value * value
    listener_handler = traits_listener.ListenerHandler(square)
    handler = listener_handler()
    self.assertEqual(handler(9), 81)
    del square, handler
    handler = listener_handler()
    self.assertEqual(handler(5), 25)