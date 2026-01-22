import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_join_with_colon(self):
    actual = parse('a:b:c')
    expected = trait('a', False).trait('b', False).trait('c')
    self.assertEqual(actual, expected)