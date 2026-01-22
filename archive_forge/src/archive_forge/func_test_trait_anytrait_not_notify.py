import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_trait_anytrait_not_notify(self):
    actual = parse('name:*')
    expected = trait('name', notify=False).anytrait()
    self.assertEqual(actual, expected)