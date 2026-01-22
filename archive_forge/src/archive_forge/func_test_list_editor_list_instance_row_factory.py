import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_list_editor_list_instance_row_factory(self):
    trait = List(Instance(HasTraits, kw={}))
    editor = trait.create_editor()
    self.assertIsInstance(editor, traitsui.api.TableEditor)
    if editor.row_factory is not None:
        self.assertTrue(callable(editor.row_factory))