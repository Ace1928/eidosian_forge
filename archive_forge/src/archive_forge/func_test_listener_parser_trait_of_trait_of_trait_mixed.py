from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_listener_parser_trait_of_trait_of_trait_mixed(self):
    text = 'parent.child1:child2'
    parser = traits_listener.ListenerParser(text=text)
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', is_list_handler=False, type=traits_listener.ANY_LISTENER)
    expected_child2 = traits_listener.ListenerItem(name='child2', notify=True, next=None, **common_traits)
    expected_child1 = traits_listener.ListenerItem(name='child1', notify=False, next=expected_child2, **common_traits)
    expected_parent = traits_listener.ListenerItem(name='parent', notify=True, next=expected_child1, **common_traits)
    self.assertEqual(parser.listener, expected_parent)