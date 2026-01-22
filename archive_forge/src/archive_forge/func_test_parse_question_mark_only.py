from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_question_mark_only(self):
    text = '?'
    with self.assertRaises(TraitError) as exception_context:
        traits_listener.ListenerParser(text=text)
    self.assertIn('Expected non-empty name', str(exception_context.exception))