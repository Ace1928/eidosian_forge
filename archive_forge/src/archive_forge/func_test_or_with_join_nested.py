import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_or_with_join_nested(self):
    actual = parse('a.b.c,d.e')
    expected = trait('a').trait('b').trait('c') | trait('d').trait('e')
    self.assertEqual(actual, expected)