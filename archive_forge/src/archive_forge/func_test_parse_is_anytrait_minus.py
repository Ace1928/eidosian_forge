from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_is_anytrait_minus(self):
    text = '-'
    parser = traits_listener.ListenerParser(text=text)
    expected = traits_listener.ListenerItem(name='*', metadata_name='', metadata_defined=False, is_anytrait=True, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    self.assertEqual(parser.listener, expected)