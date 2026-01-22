from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_square_bracket(self):
    text = '[foo, bar]'
    parser = traits_listener.ListenerParser(text=text)
    listener_group = parser.listener
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    expected_items = [traits_listener.ListenerItem(name='foo', **common_traits), traits_listener.ListenerItem(name='bar', **common_traits)]
    self.assertEqual(len(listener_group.items), len(expected_items))
    for actual, expected in zip(listener_group.items, expected_items):
        self.assertEqual(actual, expected)