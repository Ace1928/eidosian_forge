from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_is_anytrait_plus(self):
    text = '+'
    parser = traits_listener.ListenerParser(text=text)
    expected = traits_listener.ListenerItem(name='*', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    self.assertEqual(parser.listener, expected)