from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_square_bracket_in_middle(self):
    text = 'foo.[bar, baz]'
    parser = traits_listener.ListenerParser(text=text)
    actual_foo = parser.listener
    actual_next = actual_foo.next
    actual_foo.next = None
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    expected_foo = traits_listener.ListenerItem(name='foo', **common_traits)
    self.assertEqual(actual_foo, expected_foo)
    expected_items = [traits_listener.ListenerItem(name='bar', **common_traits), traits_listener.ListenerItem(name='baz', **common_traits)]
    self.assertEqual(len(actual_next.items), len(expected_items))
    for actual, expected in zip(actual_next.items, expected_items):
        self.assertEqual(actual, expected)