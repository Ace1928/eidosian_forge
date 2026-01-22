from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_comma_separated_text_trailing_comma(self):
    text = 'child1, child2, child3,'
    with self.assertRaises(TraitError):
        traits_listener.ListenerParser(text=text)