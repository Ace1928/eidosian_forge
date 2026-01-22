import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_bytes_editor_options(self):
    editor = bytes_editor(auto_set=False, enter_set=True, encoding='ascii')
    self.assertIsInstance(editor, traitsui.api.TextEditor)
    self.assertFalse(editor.auto_set)
    self.assertTrue(editor.enter_set)
    formatted = editor.format_func(b'deadbeef')
    self.assertEqual(formatted, 'deadbeef')
    evaluated = editor.evaluate('deadbeef')
    self.assertEqual(evaluated, b'deadbeef')