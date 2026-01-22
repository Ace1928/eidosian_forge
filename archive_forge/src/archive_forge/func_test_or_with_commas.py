import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_or_with_commas(self):
    actual = parse('a,b,c')
    expected = trait('a') | trait('b') | trait('c')
    self.assertEqual(actual, expected)