from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_listener_parser_trait_of_trait_dot(self):
    text = 'parent.child'
    parser = traits_listener.ListenerParser(text=text)
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER)
    expected_child = traits_listener.ListenerItem(name='child', next=None, **common_traits)
    expected_parent = traits_listener.ListenerItem(name='parent', next=expected_child, **common_traits)
    self.assertEqual(parser.listener, expected_parent)