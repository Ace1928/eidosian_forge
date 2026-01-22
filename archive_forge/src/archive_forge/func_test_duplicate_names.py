import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_duplicate_names(self):
    from traitsui.api import Group, Item, View

    class Model(HasTraits):
        count = Int
        includable = Group(Item('count', id='name_conflict'))
        name_conflict = View(Item('count'))
    with self.assertRaises(TraitError):
        Model.class_trait_view_elements()