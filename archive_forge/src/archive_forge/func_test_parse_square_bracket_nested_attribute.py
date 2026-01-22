from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_square_bracket_nested_attribute(self):
    text = '[foo, bar].baz'
    parser = traits_listener.ListenerParser(text=text)
    listener_group = parser.listener
    self.assertEqual(len(listener_group.items), 2)
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER)
    child_listener = traits_listener.ListenerItem(name='baz', next=None, **common_traits)
    expected_items = [traits_listener.ListenerItem(name='foo', next=child_listener, **common_traits), traits_listener.ListenerItem(name='bar', next=child_listener, **common_traits)]
    self.assertEqual(len(listener_group.items), len(expected_items))
    for actual, expected in zip(listener_group.items, expected_items):
        self.assertEqual(actual, expected)